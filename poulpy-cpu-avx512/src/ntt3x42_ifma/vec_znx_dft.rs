//! NTT-domain SIMD helpers for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! SIMD Garner reconstruction for the consume path.

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    kernels::{cond_sub_2q_si512, harvey_modmul_si512},
    module::handle,
    primes::{PrimeSetNtt3x42Ifma, Primes42},
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::{
        Ntt3x42IfmaAdd, Ntt3x42IfmaAddAssign, Ntt3x42IfmaCopy, Ntt3x42IfmaDFTExecute, Ntt3x42IfmaFromZnx64, Ntt3x42IfmaNegate,
        Ntt3x42IfmaNegateAssign, Ntt3x42IfmaSub, Ntt3x42IfmaSubAssign, Ntt3x42IfmaSubNegateAssign, Ntt3x42IfmaToZnx128,
        Ntt3x42IfmaZero,
    },
};
use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
};
use poulpy_hal::layouts::{
    Data, HostDataMut, HostDataRef, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut,
    VecZnxDftBackendRef, ZnxView, ZnxViewMut,
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
const Q01_LO: u64 = (Q01 & ((1u128 << 52) - 1)) as u64;
const Q01_HI: u64 = (Q01 >> 52) as u64;

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

/// CRT reconstruction: planar 3-prime IFMA b-format to i128.
///
/// Input residues must be in `[0, 2q)` (b-format after iNTT).
///
/// # Safety
///
/// - `a` must contain at least `3 * nn` u64 values.
/// - `res` must have room for at least `nn` i128 values.
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn simd_b_ntt3x42_ifma_to_znx128(nn: usize, res: &mut [i128], a: &[u64]) {
    debug_assert!(res.len() >= nn);
    debug_assert!(a.len() >= 3 * nn);

    unsafe {
        let q0 = _mm512_set1_epi64(Q0 as i64);
        let q1 = _mm512_set1_epi64(Q1 as i64);
        let q2 = _mm512_set1_epi64(Q2 as i64);
        let inv01 = _mm512_set1_epi64(INV01 as i64);
        let inv01_quot = _mm512_set1_epi64(INV01_QUOT as i64);
        let inv012 = _mm512_set1_epi64(INV012 as i64);
        let inv012_quot = _mm512_set1_epi64(INV012_QUOT as i64);
        let q0_mod_q2 = _mm512_set1_epi64(Q0_MOD_Q2 as i64);
        let q0_mod_q2_quot = _mm512_set1_epi64(Q0_MOD_Q2_QUOT as i64);
        let q01_lo = _mm512_set1_epi64(Q01_LO as i64);
        let q01_hi = _mm512_set1_epi64(Q01_HI as i64);
        let mask52 = _mm512_set1_epi64(((1u64 << 52) - 1) as i64);
        let mask12 = _mm512_set1_epi64(0xFFF);
        let zero = _mm512_setzero_si512();
        let mut lo_lanes = [0u64; 8];
        let mut hi_lanes = [0u64; 8];

        let mut c = 0usize;
        while c + 8 <= nn {
            let r0 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(c) as *const __m512i), q0);
            let r1 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(nn + c) as *const __m512i), q1);
            let r2 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(2 * nn + c) as *const __m512i), q2);

            let v0_mod_q1 = cond_sub_2q_si512(r0, q1);
            let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(r1, q1), v0_mod_q1), q1);
            let v1 = harvey_modmul_si512(diff1, inv01, inv01_quot, q1);

            let v0_mod_q2 = cond_sub_2q_si512(r0, q2);
            let v1q0_mod_q2 = harvey_modmul_si512(v1, q0_mod_q2, q0_mod_q2_quot, q2);
            let partial = cond_sub_2q_si512(_mm512_add_epi64(v0_mod_q2, v1q0_mod_q2), q2);
            let diff2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(r2, q2), partial), q2);
            let v2 = harvey_modmul_si512(diff2, inv012, inv012_quot, q2);

            // result = v0 + v1*Q0 + v2*Q01, accumulated in base-2^52 limbs.
            let p10 = _mm512_madd52lo_epu64(zero, v1, q0);
            let p1h = _mm512_madd52hi_epu64(zero, v1, q0);
            let p2ll = _mm512_madd52lo_epu64(zero, v2, q01_lo);
            let p2lh = _mm512_madd52hi_epu64(zero, v2, q01_lo);
            let p2hl = _mm512_madd52lo_epu64(zero, v2, q01_hi);
            let p2hh = _mm512_madd52hi_epu64(zero, v2, q01_hi);
            let a0 = _mm512_add_epi64(_mm512_add_epi64(r0, p10), p2ll);
            let a1 = _mm512_add_epi64(_mm512_add_epi64(p1h, p2lh), p2hl);
            let a1 = _mm512_add_epi64(a1, _mm512_srli_epi64::<52>(a0));
            let a0 = _mm512_and_si512(a0, mask52);
            let a2 = _mm512_add_epi64(p2hh, _mm512_srli_epi64::<52>(a1));
            let a1 = _mm512_and_si512(a1, mask52);
            let lo = _mm512_add_epi64(a0, _mm512_slli_epi64::<52>(_mm512_and_si512(a1, mask12)));
            let hi = _mm512_add_epi64(_mm512_srli_epi64::<12>(a1), _mm512_slli_epi64::<40>(a2));
            _mm512_storeu_si512(lo_lanes.as_mut_ptr() as *mut __m512i, lo);
            _mm512_storeu_si512(hi_lanes.as_mut_ptr() as *mut __m512i, hi);
            for lane in 0..8 {
                let result = lo_lanes[lane] as u128 | ((hi_lanes[lane] as u128) << 64);
                res[c + lane] = if result > HALF_BIG_Q {
                    result as i128 - BIG_Q as i128
                } else {
                    result as i128
                };
            }

            c += 8;
        }

        while c < nn {
            let r0 = cond_sub_scalar(a[c], Q0);
            let r1 = cond_sub_scalar(a[nn + c], Q1);
            let r2 = cond_sub_scalar(a[2 * nn + c], Q2);
            res[c] = garner_from_residues(r0, r1, r2);
            c += 1;
        }
    }
}

/// iNTT (in place on `src`) + Garner CRT-compact (writing i128 to `dst`).
///
/// # Safety
/// - `src_ptr` covers `3 * n * n_blocks` u64; `dst_ptr` covers `n * n_blocks` i128.
/// - If aliased, the dst window must lie in the first half of the src window.
/// - AVX-512-IFMA and AVX-512-VL required at runtime.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_then_compact_ifma(
    n: usize,
    n_blocks: usize,
    src_ptr: *mut u64,
    dst_ptr: *mut i128,
    table: &Ntt3x42IfmaTableInv<Primes42>,
) {
    unsafe {
        for k in 0..n_blocks {
            let src_off_u64 = 3 * n * k;
            let dst_off_i128 = n * k;

            // Step 1: inverse NTT in-place on `src`.
            {
                let blk = std::slice::from_raw_parts_mut(src_ptr.add(src_off_u64), 3 * n);
                <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>>>::ntt3x42_ifma_dft_execute(table, blk);
            }

            // Step 2: Garner CRT-compact 3n u64s → n i128s, writing to `dst`.
            let src_base = src_ptr.add(src_off_u64);
            let dst_base = dst_ptr.add(dst_off_i128);
            let src = std::slice::from_raw_parts(src_base, 3 * n);
            let dst = std::slice::from_raw_parts_mut(dst_base, n);

            simd_b_ntt3x42_ifma_to_znx128(n, dst, src);
        }
    }
}

/// `VecZnxIdftApplyTmpA` fast path: iNTT consumes `a` in place, Garner streams
/// into `res`. Limbs past `min(res.size(), a.size())` are zero-padded.
pub(crate) fn vec_znx_idft_apply_tmpa_ifma(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxBigBackendMut<'_, crate::NTT3x42Ifma>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
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
        let src_off_u64 = 3 * n * (j * a_cols + a_col);
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
fn limb_u64<D: Data + HostDataRef>(v: &VecZnxDft<D, NTT3x42Ifma>, col: usize, limb: usize) -> &[u64] {
    cast_slice(v.at(col, limb))
}

#[inline(always)]
fn limb_u64_mut<D: Data + HostDataMut>(v: &mut VecZnxDft<D, NTT3x42Ifma>, col: usize, limb: usize) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

/// Forward NTT for the IFMA backend.
pub(crate) fn vec_znx_dft_apply(
    module: &Module<NTT3x42Ifma>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
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
            NTT3x42Ifma::ntt3x42_ifma_from_znx64(res_slice, a.at(a_col, limb));
            <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42>>>::ntt3x42_ifma_dft_execute(table, res_slice);
        } else {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}

/// Scratch space (in bytes) for [`vec_znx_idft_apply`].
pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    use std::mem::size_of;
    3 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive) for the IFMA backend.
pub(crate) fn vec_znx_idft_apply(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxBigBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let table = &handle(module).table_intt;

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..3 * n];
        NTT3x42Ifma::ntt3x42_ifma_copy(tmp_n, a_slice);
        <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>>>::ntt3x42_ifma_dft_execute(table, tmp_n);
        NTT3x42Ifma::ntt3x42_ifma_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
pub(crate) fn vec_znx_dft_add_into(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_add(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT3x42Ifma::ntt3x42_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_add(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT3x42Ifma::ntt3x42_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub(crate) fn vec_znx_dft_add_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT3x42Ifma::ntt3x42_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
pub(crate) fn vec_znx_dft_add_scaled_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    a_scale: i64,
) {
    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_add_assign(limb_u64_mut(res, res_col, j + shift), limb_u64(a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_add_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub(crate) fn vec_znx_dft_sub(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_sub(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT3x42Ifma::ntt3x42_ifma_negate(limb_u64_mut(res, res_col, j), limb_u64(b, b_col, j));
        }
        for j in cpy_size..res_size {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            NTT3x42Ifma::ntt3x42_ifma_sub(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j), limb_u64(b, b_col, j));
        }
        for j in sum_size..cpy_size {
            NTT3x42Ifma::ntt3x42_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
        }
        for j in cpy_size..res_size {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub(crate) fn vec_znx_dft_sub_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        NTT3x42Ifma::ntt3x42_ifma_sub_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
pub(crate) fn vec_znx_dft_sub_negate_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        NTT3x42Ifma::ntt3x42_ifma_sub_negate_assign(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, j));
    }
    for j in sum_size..res_size {
        NTT3x42Ifma::ntt3x42_ifma_negate_assign(limb_u64_mut(res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
pub(crate) fn vec_znx_dft_copy(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
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
            NTT3x42Ifma::ntt3x42_ifma_copy(limb_u64_mut(res, res_col, j), limb_u64(a, a_col, limb));
        } else {
            NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub(crate) fn vec_znx_dft_zero(res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>, res_col: usize) {
    for j in 0..res.size() {
        NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, j));
    }
}

/// NTT3x42 automorphism on the planar layout: each limb stores three contiguous
/// `n`-element residue planes, and the NTT slot action is a pure permutation, so
/// for every output slot `i` we copy the source slot `perm[i]` within each plane.
pub(crate) fn vec_znx_dft_automorphism(
    plan: &poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    for limb in 0..min_size {
        let a_slice: &[u64] = limb_u64(a, a_col, limb);
        let res_slice: &mut [u64] = limb_u64_mut(res, res_col, limb);
        for plane in 0..3 {
            let base: usize = plane * n;
            let src: &[u64] = &a_slice[base..base + n];
            let dst: &mut [u64] = &mut res_slice[base..base + n];
            for i in 0..n {
                dst[i] = src[perm[i] as usize];
            }
        }
    }

    for limb in min_size..res_size {
        NTT3x42Ifma::ntt3x42_ifma_zero(limb_u64_mut(res, res_col, limb));
    }
}
