//! Primitive NTT-domain trait implementations for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! This module connects the IFMA backend type to the low-level reference IFMA traits:
//! NTT execution, b/c domain conversion, BBC multiply-accumulate, and basic
//! transform-domain arithmetic on the planar 3-prime prep representation.

use crate::ntt3x42_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    primes::Primes42,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaDFTExecute, Ntt3x42IfmaFromZnx64, Ntt3x42IfmaMulBbc, Ntt3x42IfmaToZnx128},
};
use poulpy_hal::layouts::PrimeSet;

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm512_add_epi64, _mm512_and_si512,
    _mm512_cmpgt_epi64_mask, _mm512_loadu_si512, _mm512_maskz_mov_epi64, _mm512_mul_epu32, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_srli_epi64, _mm512_storeu_si512,
};

use super::kernels::{cond_sub_2q_si256, cond_sub_2q_si512, intt_avx512, ntt_avx512};
use super::mat_vec_ifma::vec_mat1col_product_bbc_ifma;

use poulpy_cpu_ref::reference::ntt4x30::{
    NttAdd, NttAddAssign, NttCopy, NttNegate, NttNegateAssign, NttSub, NttSubAssign, NttSubNegateAssign, NttZero,
};

use crate::NTT3x42Ifma;

#[target_feature(enable = "avx512f")]
unsafe fn simd_add(res: &mut [u64], a: &[u64], b: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = a[base + i] + b[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_add_assign(res: &mut [u64], a: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = res[base + i] + a[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = a[base + i] + q2 - b[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_assign(res: &mut [u64], a: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = res[base + i] + q2 - a[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_negate_assign(res: &mut [u64], a: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = a[base + i] + q2 - res[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate(res: &mut [u64], a: &[u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = q2 - a[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate_assign(res: &mut [u64]) {
    let n = res.len() / 3;
    for p in 0..3 {
        let q2 = 2 * Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = q2 - res[base + i];
            res[base + i] = if x >= q2 { x - q2 } else { x };
        }
    }
}

/// `oq[k] = Q[k] - (2^63 mod Q[k])` for negative i64 handling.
const OQ_IFMA: [u64; 4] = {
    let q = <Primes42 as poulpy_hal::layouts::PrimeSet>::Q;
    let mut oq = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        oq[k] = q[k] - (i64::MIN as u64 % q[k]);
        k += 1;
    }
    oq
};

/// `2^42 mod Q[k]` for two-pass reduction of signed 64-bit inputs.
const POW42_MOD_Q_IFMA: [u64; 4] = {
    let q = <Primes42 as poulpy_hal::layouts::PrimeSet>::Q;
    let pow42 = 1u64 << 42;
    [pow42 - q[0], pow42 - q[1], pow42 - q[2], 0]
};

/// Convert n i64 coefficients to planar 3-prime CRT b format.
///
/// For each i64 x:
/// 1. Strip sign bit, conditionally add oq[k] for negative inputs
/// 2. Two-pass reduction: split at bit 42, multiply high part by (2^42 mod Q),
///    add to low part, repeat. Final conditional subtract gives [0, Q).
///
/// Result: `res[k*n+i] = a[i] mod Q[k]` for k in {0,1,2}.
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64(n: usize, res: &mut [u64], a: &[i64]) {
    unsafe { simd_b_from_znx64_impl(n, res, a, !0i64) }
}

/// Same as [`simd_b_from_znx64`] but ANDs each input by `mask` first.
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64_masked(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    unsafe { simd_b_from_znx64_impl(n, res, a, mask) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn simd_b_from_znx64_impl(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    debug_assert!(res.len() >= 3 * n);
    debug_assert!(a.len() >= n);
    unsafe {
        let oq_vec = _mm256_loadu_si256(OQ_IFMA.as_ptr() as *const __m256i);
        let i64_max = _mm256_set1_epi64x(i64::MAX);
        let zero = _mm256_setzero_si256();
        let mask42 = _mm256_set1_epi64x((1i64 << 42) - 1);
        let pow42 = _mm256_loadu_si256(POW42_MOD_Q_IFMA.as_ptr() as *const __m256i);
        let q = _mm256_loadu_si256(Primes42::Q.as_ptr() as *const __m256i);
        let mask_vec = _mm256_set1_epi64x(mask);
        let mut lanes = [0u64; 4];
        let lanes_ptr = lanes.as_mut_ptr() as *mut __m256i;

        let i64_max512 = _mm512_set1_epi64(i64::MAX);
        let zero512 = _mm512_setzero_si512();
        let mask42_512 = _mm512_set1_epi64((1i64 << 42) - 1);
        let mask512 = _mm512_set1_epi64(mask);
        let oq0_512 = _mm512_set1_epi64(OQ_IFMA[0] as i64);
        let oq1_512 = _mm512_set1_epi64(OQ_IFMA[1] as i64);
        let oq2_512 = _mm512_set1_epi64(OQ_IFMA[2] as i64);
        let pow42_0_512 = _mm512_set1_epi64(POW42_MOD_Q_IFMA[0] as i64);
        let pow42_1_512 = _mm512_set1_epi64(POW42_MOD_Q_IFMA[1] as i64);
        let pow42_2_512 = _mm512_set1_epi64(POW42_MOD_Q_IFMA[2] as i64);
        let q0_512 = _mm512_set1_epi64(Primes42::Q[0] as i64);
        let q1_512 = _mm512_set1_epi64(Primes42::Q[1] as i64);
        let q2_512 = _mm512_set1_epi64(Primes42::Q[2] as i64);
        let mut i = 0usize;
        while i + 8 <= n {
            let xv = _mm512_and_si512(_mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i), mask512);
            let xl = _mm512_and_si512(xv, i64_max512);
            let sign = _mm512_cmpgt_epi64_mask(zero512, xv);

            let reduce = |oq: __m512i, pow42: __m512i, q: __m512i| {
                let val = _mm512_add_epi64(xl, _mm512_maskz_mov_epi64(sign, oq));
                let hi = _mm512_srli_epi64::<42>(val);
                let lo = _mm512_and_si512(val, mask42_512);
                let y = _mm512_add_epi64(_mm512_mul_epu32(hi, pow42), lo);

                let hi2 = _mm512_srli_epi64::<42>(y);
                let lo2 = _mm512_and_si512(y, mask42_512);
                let z = _mm512_add_epi64(_mm512_mul_epu32(hi2, pow42), lo2);
                cond_sub_2q_si512(z, q)
            };

            _mm512_storeu_si512(res.as_mut_ptr().add(i) as *mut __m512i, reduce(oq0_512, pow42_0_512, q0_512));
            _mm512_storeu_si512(
                res.as_mut_ptr().add(n + i) as *mut __m512i,
                reduce(oq1_512, pow42_1_512, q1_512),
            );
            _mm512_storeu_si512(
                res.as_mut_ptr().add(2 * n + i) as *mut __m512i,
                reduce(oq2_512, pow42_2_512, q2_512),
            );
            i += 8;
        }

        while i < n {
            let xv = _mm256_and_si256(_mm256_set1_epi64x(a[i]), mask_vec);
            let xl = _mm256_and_si256(xv, i64_max);
            let sign = _mm256_cmpgt_epi64(zero, xv);
            let add = _mm256_and_si256(sign, oq_vec);
            let val = _mm256_add_epi64(xl, add);

            let hi = _mm256_srli_epi64::<42>(val);
            let lo = _mm256_and_si256(val, mask42);
            let y = _mm256_add_epi64(_mm256_mul_epu32(hi, pow42), lo);

            let hi2 = _mm256_srli_epi64::<42>(y);
            let lo2 = _mm256_and_si256(y, mask42);
            let z = _mm256_add_epi64(_mm256_mul_epu32(hi2, pow42), lo2);

            _mm256_storeu_si256(lanes_ptr, cond_sub_2q_si256(z, q));
            res[i] = lanes[0];
            res[n + i] = lanes[1];
            res[2 * n + i] = lanes[2];
            i += 1;
        }
    }
}

/// Reduce planar b-format values in [0, 4q) to planar c-format values in [0, q).
#[target_feature(enable = "avx512f")]
unsafe fn simd_c_from_b(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        for p in 0..3 {
            let q = _mm512_set1_epi64(Primes42::Q[p] as i64);
            let q2 = _mm512_set1_epi64((2 * Primes42::Q[p]) as i64);
            let base = p * n;
            let mut i = 0usize;
            while i + 8 <= n {
                // [0, 4q) -> [0, q): subtract 2q then q.
                let av = _mm512_loadu_si512(a.as_ptr().add(base + i) as *const __m512i);
                let r = cond_sub_2q_si512(cond_sub_2q_si512(av, q2), q);
                _mm512_storeu_si512(res.as_mut_ptr().add(base + i) as *mut __m512i, r);
                i += 8;
            }
            while i < n {
                let mut x = a[base + i];
                if x >= 2 * Primes42::Q[p] {
                    x -= 2 * Primes42::Q[p];
                }
                res[base + i] = if x >= Primes42::Q[p] { x - Primes42::Q[p] } else { x };
                i += 1;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTable<Primes42>, data: &mut [u64]) {
        // Non-lazy: fully reduce for the public DFT contract.
        unsafe { ntt_avx512::<Primes42>(table, data, false) }
    }
}

impl Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>> for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_dft_execute(table: &Ntt3x42IfmaTableInv<Primes42>, data: &mut [u64]) {
        unsafe { intt_avx512::<Primes42>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt3x42IfmaFromZnx64 for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_from_znx64(res: &mut [u64], a: &[i64]) {
        unsafe { simd_b_from_znx64(a.len(), res, a) };
    }

    #[inline(always)]
    fn ntt3x42_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        unsafe { simd_b_from_znx64_masked(a.len(), res, a, mask) };
    }
}

impl Ntt3x42IfmaToZnx128 for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        unsafe { super::vec_znx_dft::simd_b_ntt3x42_ifma_to_znx128(divisor_is_n, res, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt3x42IfmaMulBbc for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_mul_bbc(meta: &Bbc126IfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        unsafe { vec_mat1col_product_bbc_ifma(meta, ell, res, ntt_coeff, prepared) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// b -> c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt3x42IfmaCFromB for NTT3x42Ifma {
    #[inline(always)]
    fn ntt3x42_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // c format for IFMA = reduced residues: a[k] mod Q[k].
        // Values in [0, 2q) → cond_sub with q → [0, q).
        // res is typed as &mut [u32] for trait compatibility but is actually u64 data.
        let res_u64: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u64, res.len() / 2) };
        unsafe { simd_c_from_b(n, res_u64, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT4x30 Ntt* traits (DFT-domain arithmetic reuse)
//
// The ntt4x30 generic functions (ntt4x30_vec_znx_dft_add, etc.) require these
// traits. Since the 3-prime IFMA layout uses the same 4 x u64 per coefficient
// representation (with lane 3 as padding), the implementation is identical
// to the NTT4x30 version but uses Q_SHIFTED_NTT3X42IFMA for the 3 active lanes.
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl NttAddAssign for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_assign(res, a) };
    }
}

impl NttSub for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl NttSubAssign for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_assign(res, a) };
    }
}

impl NttSubNegateAssign for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_assign(res, a) };
    }
}

impl NttNegate for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl NttNegateAssign for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        unsafe { simd_negate_assign(res) };
    }
}

impl NttZero for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT3x42Ifma {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
