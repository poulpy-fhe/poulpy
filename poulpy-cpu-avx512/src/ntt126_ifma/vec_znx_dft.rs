//! NTT-domain SIMD helpers for [`NTT126Ifma`](crate::NTT126Ifma).
//!
//! SIMD Garner reconstruction for the consume path.

use crate::NTT126Ifma;
use crate::ntt126_ifma::{
    module::handle,
    primes::{PrimeSetNtt126Ifma, Primes42},
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
    traits::{
        Ntt126IfmaAdd, Ntt126IfmaAddAssign, Ntt126IfmaCopy, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64, Ntt126IfmaNegate,
        Ntt126IfmaNegateAssign, Ntt126IfmaSub, Ntt126IfmaSubAssign, Ntt126IfmaSubNegateAssign, Ntt126IfmaToZnx128,
        Ntt126IfmaZero,
    },
};
use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::{
    Data, DataMut, DataRef, Module, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos,
    ZnxView, ZnxViewMut,
};

use super::kernels::{cond_sub_2q_si256, harvey_modmul_si256, intt_avx512};

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_set_epi64x, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
};

// 3-prime CRT -> i128 reconstruction helpers.

const Q: [u64; 3] = Primes42::Q;
const INV01: u64 = Primes42::CRT_CST[0];
const INV012: u64 = Primes42::CRT_CST[1];
const Q0: u64 = Q[0];
const Q1: u64 = Q[1];
const Q2: u64 = Q[2];
const Q01: u128 = Q0 as u128 * Q1 as u128;
const BIG_Q: u128 = Q01 * Q2 as u128;
const HALF_BIG_Q: u128 = BIG_Q / 2;

// Harvey quotients for the Garner steps.
const INV01_QUOT: u64 = ((INV01 as u128 * (1u128 << 52)) / Q1 as u128) as u64;
const INV012_QUOT: u64 = ((INV012 as u128 * (1u128 << 52)) / Q2 as u128) as u64;
// `Q0 mod Q2` and its Harvey quotient.
const Q0_MOD_Q2: u64 = Q0 % Q2;
const Q0_MOD_Q2_QUOT: u64 = ((Q0_MOD_Q2 as u128 * (1u128 << 52)) / Q2 as u128) as u64;

/// Harvey scalar modular multiply: `(a * omega) mod q`, result in `[0, q)`.
///
/// Input: `a ∈ [0, q)`, `omega ∈ [0, q)`.
/// `omega_quot = floor(omega * 2^52 / q)`.
#[inline(always)]
fn harvey_modmul_scalar(a: u64, omega: u64, omega_quot: u64, q: u64) -> u64 {
    let qhat = ((a as u128 * omega_quot as u128) >> 52) as u64;
    let product_lo = (a as u128 * omega as u128) as u64;
    let qhat_times_q = (qhat as u128 * q as u128) as u64;
    let mut r = product_lo.wrapping_sub(qhat_times_q);
    if (r as i64) < 0 {
        r = r.wrapping_add(q);
    }
    if r >= q { r - q } else { r }
}

/// Conditional subtract: if x >= q, return x - q.
#[inline(always)]
fn cond_sub_scalar(x: u64, q: u64) -> u64 {
    if x >= q { x - q } else { x }
}

/// SIMD-assisted single-coefficient Garner CRT reconstruction.
///
/// Reduces one packed residue vector to `[0, q)` and reconstructs one `i128`.
///
/// # Safety
///
/// - `src` must be valid for reading 4 × u64 (one `__m256i`).
/// - Caller must ensure AVX512-VL support.
#[target_feature(enable = "avx512vl")]
pub(crate) unsafe fn garner_crt_single(src: *const u64, q_vec: __m256i) -> i128 {
    unsafe {
        let xv = _mm256_loadu_si256(src as *const __m256i);
        let reduced = cond_sub_2q_si256(xv, q_vec);

        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, reduced);
        let (r0, r1, r2) = (lanes[0], lanes[1], lanes[2]);

        garner_from_residues(r0, r1, r2)
    }
}

/// Scalar Garner CRT reconstruction from 3 reduced residues.
///
/// Input: `r0 ∈ [0, Q0)`, `r1 ∈ [0, Q1)`, `r2 ∈ [0, Q2)`.
/// Output: reconstructed `i128` in symmetric representation `(-Q/2, Q/2]`.
#[inline(always)]
fn garner_from_residues(r0: u64, r1: u64, r2: u64) -> i128 {
    let v0 = r0;

    let v0_mod_q1 = cond_sub_scalar(v0, Q1);
    let diff1 = cond_sub_scalar(r1 + Q1 - v0_mod_q1, Q1);
    let v1 = harvey_modmul_scalar(diff1, INV01, INV01_QUOT, Q1);

    let v0_mod_q2 = cond_sub_scalar(v0, Q2);
    let v1q0_mod_q2 = harvey_modmul_scalar(v1, Q0_MOD_Q2, Q0_MOD_Q2_QUOT, Q2);
    let partial = cond_sub_scalar(v0_mod_q2 + v1q0_mod_q2, Q2);
    let diff2 = cond_sub_scalar(r2 + Q2 - partial, Q2);
    let v2 = harvey_modmul_scalar(diff2, INV012, INV012_QUOT, Q2);

    let result_u128 = v0 as u128 + v1 as u128 * Q0 as u128 + v2 as u128 * Q01;

    if result_u128 > HALF_BIG_Q {
        result_u128 as i128 - BIG_Q as i128
    } else {
        result_u128 as i128
    }
}

/// Vectorized Garner CRT reconstruction for 4 coefficients in parallel.
///
/// Reconstructs 4 coefficients from AoS-packed residues.
///
/// # Safety
///
/// - `dst` must be valid for writing 4 × i128 (64 bytes).
/// - All input vectors must have residues in `[0, q)` (already reduced).
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn garner_4coeffs_simd(
    a0: __m256i,
    a1: __m256i,
    a2: __m256i,
    a3: __m256i,
    dst: *mut i128,
    q1_bcast: __m256i,
    q2_bcast: __m256i,
    q2x2_bcast: __m256i,
    inv01_bcast: __m256i,
    inv01_quot_bcast: __m256i,
    inv012_bcast: __m256i,
    inv012_quot_bcast: __m256i,
    q0modq2_bcast: __m256i,
    q0modq2_quot_bcast: __m256i,
) {
    unsafe {
        // Transpose AoS residues into per-prime vectors.
        let ab_lo = _mm256_unpacklo_epi64(a0, a1); // [a.r0, b.r0, a.r2, b.r2]
        let ab_hi = _mm256_unpackhi_epi64(a0, a1); // [a.r1, b.r1, 0, 0]
        let cd_lo = _mm256_unpacklo_epi64(a2, a3); // [c.r0, d.r0, c.r2, d.r2]
        let cd_hi = _mm256_unpackhi_epi64(a2, a3); // [c.r1, d.r1, 0, 0]

        let vec_r0 = _mm256_permute2x128_si256::<0x20>(ab_lo, cd_lo); // [a.r0, b.r0, c.r0, d.r0]
        let vec_r1 = _mm256_permute2x128_si256::<0x20>(ab_hi, cd_hi); // [a.r1, b.r1, c.r1, d.r1]
        let vec_r2 = _mm256_permute2x128_si256::<0x31>(ab_lo, cd_lo); // [a.r2, b.r2, c.r2, d.r2]

        let vec_v0 = vec_r0;

        // Step 2: recover `v1`.
        let v0_mod_q1 = cond_sub_2q_si256(vec_v0, q1_bcast);
        let diff1_raw = _mm256_sub_epi64(_mm256_add_epi64(vec_r1, q1_bcast), v0_mod_q1);
        let diff1 = cond_sub_2q_si256(diff1_raw, q1_bcast);
        let vec_v1_lazy = harvey_modmul_si256(diff1, inv01_bcast, inv01_quot_bcast, q1_bcast);
        let vec_v1 = cond_sub_2q_si256(vec_v1_lazy, q1_bcast);

        // Step 3: recover `v2`.
        let v0_mod_q2 = cond_sub_2q_si256(vec_v0, q2_bcast);
        let v1q0_lazy = harvey_modmul_si256(vec_v1, q0modq2_bcast, q0modq2_quot_bcast, q2_bcast);
        let partial_raw = _mm256_add_epi64(v0_mod_q2, v1q0_lazy);
        let partial = cond_sub_2q_si256(cond_sub_2q_si256(partial_raw, q2x2_bcast), q2_bcast);
        let diff2_raw = _mm256_sub_epi64(_mm256_add_epi64(vec_r2, q2_bcast), partial);
        let diff2 = cond_sub_2q_si256(diff2_raw, q2_bcast);
        let vec_v2_lazy = harvey_modmul_si256(diff2, inv012_bcast, inv012_quot_bcast, q2_bcast);
        let vec_v2 = cond_sub_2q_si256(vec_v2_lazy, q2_bcast);

        // Recombine the Garner digits into signed `i128`.
        let mut v0s = [0u64; 4];
        let mut v1s = [0u64; 4];
        let mut v2s = [0u64; 4];
        _mm256_storeu_si256(v0s.as_mut_ptr() as *mut __m256i, vec_v0);
        _mm256_storeu_si256(v1s.as_mut_ptr() as *mut __m256i, vec_v1);
        _mm256_storeu_si256(v2s.as_mut_ptr() as *mut __m256i, vec_v2);

        for lane in 0..4 {
            let result_u128 = v0s[lane] as u128 + v1s[lane] as u128 * Q0 as u128 + v2s[lane] as u128 * Q01;
            let val: i128 = if result_u128 > HALF_BIG_Q {
                result_u128 as i128 - BIG_Q as i128
            } else {
                result_u128 as i128
            };
            dst.add(lane).write_unaligned(val);
        }
    }
}

/// Vectorized CRT reconstruction: 3-prime IFMA b-format to i128.
///
/// Processes coefficients in batches of 4 using SIMD Garner reconstruction.
/// Falls back to single-coefficient path for the tail.
///
/// Input residues must be in `[0, 2q)` (b-format after iNTT).
///
/// # Safety
///
/// - `a` must contain at least `4 * nn` u64 values.
/// - `res` must have room for at least `nn` i128 values.
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn simd_b_ntt126_ifma_to_znx128(nn: usize, res: &mut [i128], a: &[u64]) {
    unsafe {
        // Per-prime constants for the scalar and SIMD paths.
        let q_vec = _mm256_set_epi64x(0, Q2 as i64, Q1 as i64, Q0 as i64);
        let q1_bcast = _mm256_set1_epi64x(Q1 as i64);
        let q2_bcast = _mm256_set1_epi64x(Q2 as i64);
        let q2x2_bcast = _mm256_set1_epi64x((2 * Q2) as i64);
        let inv01_bcast = _mm256_set1_epi64x(INV01 as i64);
        let inv01_quot_bcast = _mm256_set1_epi64x(INV01_QUOT as i64);
        let inv012_bcast = _mm256_set1_epi64x(INV012 as i64);
        let inv012_quot_bcast = _mm256_set1_epi64x(INV012_QUOT as i64);
        let q0modq2_bcast = _mm256_set1_epi64x(Q0_MOD_Q2 as i64);
        let q0modq2_quot_bcast = _mm256_set1_epi64x(Q0_MOD_Q2_QUOT as i64);

        let a_ptr = a.as_ptr() as *const __m256i;
        let dst = res.as_mut_ptr();

        // Main loop: 4 coefficients at a time
        let mut c = 0usize;
        while c + 4 <= nn {
            let a0 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c)), q_vec);
            let a1 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 1)), q_vec);
            let a2 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 2)), q_vec);
            let a3 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 3)), q_vec);

            garner_4coeffs_simd(
                a0,
                a1,
                a2,
                a3,
                dst.add(c),
                q1_bcast,
                q2_bcast,
                q2x2_bcast,
                inv01_bcast,
                inv01_quot_bcast,
                inv012_bcast,
                inv012_quot_bcast,
                q0modq2_bcast,
                q0modq2_quot_bcast,
            );
            c += 4;
        }

        // Tail: remaining coefficients (0-3)
        while c < nn {
            res[c] = garner_crt_single(a.as_ptr().add(4 * c), q_vec);
            c += 1;
        }
    }
}

/// In-place CRT-compact all NTT blocks from Q120b (32 bytes/coeff) to i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT to the 3-prime CRT block in-place.
/// 2. Uses vectorized Garner reconstruction (4 coefficients at a time via SoA
///    transpose) with Harvey modular multiply to convert residues to i128.
///
/// # Ordering invariant
///
/// Blocks are processed in order `k = 0, 1, ..., n_blocks-1`.  For `k >= 1` the
/// destination `[16nk, 16n(k+1))` never overlaps the source `[32nk, 32n(k+1))`.
/// For `k = 0` all three residues are read before the i128 is written.
///
/// # Safety
///
/// - `u64_ptr` must be valid for reads and writes of at least `4 * n * n_blocks` u64 values.
/// - The backing allocation must be at least 16-byte aligned (guaranteed by `DEFAULTALIGN = 64`).
/// - No other references to the same memory may be live during this call.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &Ntt126IfmaTableInv<Primes42>) {
    unsafe {
        // Per-prime Q vector for AoS cond_sub: [Q0, Q1, Q2, 0]
        let q_vec = _mm256_set_epi64x(0, Q2 as i64, Q1 as i64, Q0 as i64);
        // Broadcast constants for SoA Garner (loaded once)
        let q1_bcast = _mm256_set1_epi64x(Q1 as i64);
        let q2_bcast = _mm256_set1_epi64x(Q2 as i64);
        let q2x2_bcast = _mm256_set1_epi64x((2 * Q2) as i64);
        let inv01_bcast = _mm256_set1_epi64x(INV01 as i64);
        let inv01_quot_bcast = _mm256_set1_epi64x(INV01_QUOT as i64);
        let inv012_bcast = _mm256_set1_epi64x(INV012 as i64);
        let inv012_quot_bcast = _mm256_set1_epi64x(INV012_QUOT as i64);
        let q0modq2_bcast = _mm256_set1_epi64x(Q0_MOD_Q2 as i64);
        let q0modq2_quot_bcast = _mm256_set1_epi64x(Q0_MOD_Q2_QUOT as i64);

        for k in 0..n_blocks {
            let src_start = 4 * n * k;
            let dst_start = 2 * n * k;

            // Step 1: inverse NTT in-place.
            {
                let blk = std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n);
                intt_avx512::<Primes42>(table, blk);
            }

            // Step 2: Garner CRT-compact 4n u64s → n i128s.
            let src_base = u64_ptr.add(src_start) as *const __m256i;
            let dst_base = u64_ptr.add(dst_start) as *mut i128;

            // Main loop: 4 coefficients at a time via SoA Garner
            let mut c = 0usize;
            while c + 4 <= n {
                // Load 4 AoS coefficients and reduce [0, 2q) -> [0, q)
                let a0 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c)), q_vec);
                let a1 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 1)), q_vec);
                let a2 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 2)), q_vec);
                let a3 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 3)), q_vec);

                garner_4coeffs_simd(
                    a0,
                    a1,
                    a2,
                    a3,
                    dst_base.add(c),
                    q1_bcast,
                    q2_bcast,
                    q2x2_bcast,
                    inv01_bcast,
                    inv01_quot_bcast,
                    inv012_bcast,
                    inv012_quot_bcast,
                    q0modq2_bcast,
                    q0modq2_quot_bcast,
                );
                c += 4;
            }

            // Tail: remaining 0-3 coefficients via single-coefficient path
            while c < n {
                let val = garner_crt_single(u64_ptr.add(src_start + 4 * c), q_vec);
                dst_base.add(c).write_unaligned(val);
                c += 1;
            }
        }
    }
}

/// AVX512-accelerated `vec_znx_idft_apply_consume` for [`NTT126Ifma`](crate::NTT126Ifma).
///
/// Converts the DFT-domain `VecZnxDft` into a `VecZnxBig` by applying inverse NTT
/// and in-place CRT compaction (prep scalar 32 bytes/coeff → i128 16 bytes/coeff)
/// for each block, then reinterpreting the buffer.
pub(crate) fn vec_znx_idft_apply_consume<D: Data>(
    module: &Module<crate::NTT126Ifma>,
    mut a: VecZnxDft<D, crate::NTT126Ifma>,
) -> VecZnxBig<D, crate::NTT126Ifma>
where
    VecZnxDft<D, crate::NTT126Ifma>: VecZnxDftToMut<crate::NTT126Ifma>,
{
    let table = &handle(module).table_intt;

    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], crate::NTT126Ifma> = a.to_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };

    unsafe { compact_all_blocks(n, n_blocks, u64_ptr, table) };
    a.into_big()
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT-domain VecZnxDft operations
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn limb_u64<D: DataRef>(v: &VecZnxDft<D, NTT126Ifma>, col: usize, limb: usize) -> &[u64] {
    cast_slice(v.at(col, limb))
}

#[inline(always)]
fn limb_u64_mut<D: DataMut>(v: &mut VecZnxDft<D, NTT126Ifma>, col: usize, limb: usize) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

/// Forward NTT for the IFMA backend.
pub(crate) fn vec_znx_dft_apply<R, A>(
    module: &Module<NTT126Ifma>,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a = a.to_ref();

    let a_size = a.size();
    let res_size = res.size();
    let table = &handle(module).table_ntt;

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(&mut res, res_col, j);
            NTT126Ifma::ntt126_ifma_from_znx64(res_slice, a.at(a_col, limb));
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, res_slice);
        } else {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

/// Scratch space (in bytes) for [`vec_znx_idft_apply`].
pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    use std::mem::size_of;
    4 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive) for the IFMA backend.
pub(crate) fn vec_znx_idft_apply<R, A>(
    module: &Module<NTT126Ifma>,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    tmp: &mut [u64],
) where
    R: VecZnxBigToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxBig<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let table = &handle(module).table_intt;

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..4 * n];
        NTT126Ifma::ntt126_ifma_copy(tmp_n, a_slice);
        <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTableInv<Primes42>>>::ntt126_ifma_dft_execute(table, tmp_n);
        NTT126Ifma::ntt126_ifma_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// Inverse NTT (destructive — modifies `a` in place).
pub(crate) fn vec_znx_idft_apply_tmpa<R, A>(module: &Module<NTT126Ifma>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
where
    R: VecZnxBigToMut<NTT126Ifma>,
    A: VecZnxDftToMut<NTT126Ifma>,
{
    let mut res: VecZnxBig<&mut [u8], NTT126Ifma> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], NTT126Ifma> = a.to_mut();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let table = &handle(module).table_intt;

    for j in 0..min_size {
        <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTableInv<Primes42>>>::ntt126_ifma_dft_execute(
            table,
            limb_u64_mut(&mut a, a_col, j),
        );
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        NTT126Ifma::ntt126_ifma_to_znx128(res.at_mut(res_col, j), n, a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
pub(crate) fn vec_znx_dft_add_into<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
    B: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();
    let b: VecZnxDft<&[u8], NTT126Ifma> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub(crate) fn vec_znx_dft_add_assign<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
pub(crate) fn vec_znx_dft_add_scaled_assign<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(&mut res, res_col, j + shift), limb_u64(&a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub(crate) fn vec_znx_dft_sub<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
    B: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();
    let b: VecZnxDft<&[u8], NTT126Ifma> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_negate(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub(crate) fn vec_znx_dft_sub_assign<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_sub_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
pub(crate) fn vec_znx_dft_sub_negate_assign<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_sub_negate_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
    for j in sum_size..res_size {
        NTT126Ifma::ntt126_ifma_negate_assign(limb_u64_mut(&mut res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
pub(crate) fn vec_znx_dft_copy<R, A>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
    A: VecZnxDftToRef<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    let a: VecZnxDft<&[u8], NTT126Ifma> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, limb));
        } else {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub(crate) fn vec_znx_dft_zero<R>(res: &mut R, res_col: usize)
where
    R: VecZnxDftToMut<NTT126Ifma>,
{
    let mut res: VecZnxDft<&mut [u8], NTT126Ifma> = res.to_mut();
    for j in 0..res.size() {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}
