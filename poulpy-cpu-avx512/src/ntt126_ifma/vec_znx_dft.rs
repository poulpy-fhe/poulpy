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
    Data, HostDataMut, HostDataRef, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut,
    VecZnxDftBackendRef, ZnxView, ZnxViewMut,
};

use super::kernels::{cond_sub_2q_si256, intt_avx512};

use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_set_epi64x, _mm256_storeu_si256};

use std::arch::global_asm;

global_asm!(include_str!("vec_znx_dft_asm.s"), options(att_syntax));

unsafe extern "C" {
    fn ntt126_ifma_b_to_znx128_asm(nn: usize, dst: *mut i128, src: *const u64);
}

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
/// - Caller must ensure AVX512-IFMA, AVX512-VL, BMI2, and ADX support.
#[target_feature(enable = "avx512ifma,avx512vl,bmi2,adx")]
pub(crate) unsafe fn simd_b_ntt126_ifma_to_znx128(nn: usize, res: &mut [i128], a: &[u64]) {
    unsafe {
        let q_vec = _mm256_set_epi64x(0, Q2 as i64, Q1 as i64, Q0 as i64);
        let dst = res.as_mut_ptr();

        let bulk = nn & !3;
        ntt126_ifma_b_to_znx128_asm(bulk, dst, a.as_ptr());
        let mut c = bulk;

        // Tail: remaining coefficients (0-3)
        while c < nn {
            res[c] = garner_crt_single(a.as_ptr().add(4 * c), q_vec);
            c += 1;
        }
    }
}

/// iNTT (in place on `src`) + Garner CRT-compact (writing i128 to `dst`).
///
/// # Safety
/// - `src_ptr` covers `4 * n * n_blocks` u64; `dst_ptr` covers `n * n_blocks` i128.
/// - If aliased, the dst window must lie in the first half of the src window.
/// - AVX-512-IFMA, AVX-512-VL, BMI2 and ADX required at runtime.
#[target_feature(enable = "avx512ifma,avx512vl,bmi2,adx")]
unsafe fn intt_then_compact_ifma(
    n: usize,
    n_blocks: usize,
    src_ptr: *mut u64,
    dst_ptr: *mut i128,
    table: &Ntt126IfmaTableInv<Primes42>,
) {
    unsafe {
        for k in 0..n_blocks {
            let src_off_u64 = 4 * n * k;
            let dst_off_i128 = n * k;

            // Step 1: inverse NTT in-place on `src`.
            {
                let blk = std::slice::from_raw_parts_mut(src_ptr.add(src_off_u64), 4 * n);
                intt_avx512::<Primes42>(table, blk);
            }

            // Step 2: Garner CRT-compact 4n u64s → n i128s, writing to `dst`.
            let src_base = src_ptr.add(src_off_u64);
            let dst_base = dst_ptr.add(dst_off_i128);

            debug_assert!(n.is_multiple_of(4));
            ntt126_ifma_b_to_znx128_asm(n, dst_base, src_base);
        }
    }
}

/// `VecZnxIdftApplyTmpA` fast path: iNTT consumes `a` in place, Garner streams
/// into `res`. Limbs past `min(res.size(), a.size())` are zero-padded.
pub(crate) fn vec_znx_idft_apply_tmpa_ifma(
    module: &Module<crate::NTT126Ifma>,
    res: &mut VecZnxBigBackendMut<'_, crate::NTT126Ifma>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, crate::NTT126Ifma>,
    a_col: usize,
) {
    let table = &handle(module).table_intt;
    let n = a.n();
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let res_cols = res.cols();
    let res_size = res.size();

    let src_base: *mut u64 = cast_slice_mut::<_, u64>(a.raw_mut()).as_mut_ptr();
    let dst_base: *mut i128 = res.raw_mut().as_mut_ptr();

    for j in 0..min_size {
        let src_off_u64 = 4 * n * (j * a_cols + a_col);
        let dst_off_i128 = n * (j * res_cols + res_col);
        unsafe {
            intt_then_compact_ifma(n, 1, src_base.add(src_off_u64), dst_base.add(dst_off_i128), table);
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT-domain VecZnxDft operations
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn limb_u64<D: Data + HostDataRef>(v: &VecZnxDft<D, NTT126Ifma>, col: usize, limb: usize) -> &[u64] {
    cast_slice(v.at(col, limb))
}

#[inline(always)]
fn limb_u64_mut<D: Data + HostDataMut>(v: &mut VecZnxDft<D, NTT126Ifma>, col: usize, limb: usize) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

/// Forward NTT for the IFMA backend.
pub(crate) fn vec_znx_dft_apply(
    module: &Module<NTT126Ifma>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let a_size = a.size();
    let res_size = res.size();
    let table = &handle(module).table_ntt;

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(res, res_col, j);
            NTT126Ifma::ntt126_ifma_from_znx64(res_slice, a.at(a_col, limb));
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, res_slice);
        } else {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}

/// Scratch space (in bytes) for [`vec_znx_idft_apply`].
pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    use std::mem::size_of;
    4 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive) for the IFMA backend.
pub(crate) fn vec_znx_idft_apply(
    module: &Module<NTT126Ifma>,
    res: &mut VecZnxBigBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let table = &handle(module).table_intt;

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..4 * n];
        NTT126Ifma::ntt126_ifma_copy(tmp_n, a_slice);
        <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTableInv<Primes42>>>::ntt126_ifma_dft_execute(table, tmp_n);
        NTT126Ifma::ntt126_ifma_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
pub(crate) fn vec_znx_dft_add_into(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    b_col: usize,
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub(crate) fn vec_znx_dft_add_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
pub(crate) fn vec_znx_dft_add_scaled_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    a_scale: i64,
) {
    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(res, res_col, j + shift), limb_u64(a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub(crate) fn vec_znx_dft_sub(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    b_col: usize,
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_sub(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_negate(limb_u64_mut(res, res_col, j), limb_u64(b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT126Ifma::ntt126_ifma_sub(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub(crate) fn vec_znx_dft_sub_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_sub_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
pub(crate) fn vec_znx_dft_sub_negate_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        NTT126Ifma::ntt126_ifma_sub_negate_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
    for j in sum_size..res_size {
        NTT126Ifma::ntt126_ifma_negate_assign(limb_u64_mut(res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
pub(crate) fn vec_znx_dft_copy(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT126Ifma>,
    a_col: usize,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            NTT126Ifma::ntt126_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, limb));
        } else {
            NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub(crate) fn vec_znx_dft_zero(res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>, res_col: usize) {
    for j in 0..res.size() {
        NTT126Ifma::ntt126_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}
