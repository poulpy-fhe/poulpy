//! Primitive NTT-domain trait implementations for [`NTT126Ifma`](crate::NTT126Ifma).
//!
//! This module connects the IFMA backend type to the low-level reference IFMA traits:
//! NTT execution, b/c domain conversion, BBC multiply-accumulate, and basic
//! transform-domain arithmetic on the planar 3-prime prep representation.

use crate::ntt126_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    primes::{PrimeSetNtt126Ifma, Primes42},
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
    traits::{
        Ntt126IfmaAdd, Ntt126IfmaAddAssign, Ntt126IfmaCFromB, Ntt126IfmaCopy, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64,
        Ntt126IfmaMulBbc, Ntt126IfmaNegate, Ntt126IfmaNegateAssign, Ntt126IfmaSub, Ntt126IfmaSubAssign,
        Ntt126IfmaSubNegateAssign, Ntt126IfmaToZnx128, Ntt126IfmaZero,
    },
};

use super::mat_vec_ifma::vec_mat1col_product_bbc_ifma;

use poulpy_cpu_ref::reference::ntt120::{
    NttAdd, NttAddAssign, NttCopy, NttNegate, NttNegateAssign, NttSub, NttSubAssign, NttSubNegateAssign, NttZero,
};

use crate::NTT126Ifma;

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
    let q = <Primes42 as crate::ntt126_ifma::primes::PrimeSetNtt126Ifma>::Q;
    let mut oq = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        oq[k] = q[k] - (i64::MIN as u64 % q[k]);
        k += 1;
    }
    oq
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
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64_impl(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    debug_assert!(res.len() >= 3 * n);
    debug_assert!(a.len() >= n);
    for i in 0..n {
        let x = a[i] & mask;
        for p in 0..3 {
            let q = Primes42::Q[p];
            res[p * n + i] = if x >= 0 {
                (x as u64) % q
            } else {
                let pos = (x as u64) & (i64::MAX as u64);
                (pos + OQ_IFMA[p]) % q
            };
        }
    }
}

/// Reduce planar b-format values in [0, 2q) to planar c-format values in [0, q).
#[target_feature(enable = "avx512f")]
unsafe fn simd_c_from_b(n: usize, res: &mut [u64], a: &[u64]) {
    for p in 0..3 {
        let q = Primes42::Q[p];
        let base = p * n;
        for i in 0..n {
            let x = a[base + i];
            res[base + i] = if x >= q { x - q } else { x };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>> for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_dft_execute(table: &Ntt126IfmaTable<Primes42>, data: &mut [u64]) {
        crate::ntt126_ifma::reference::ntt::ntt126_ifma_ref::<Primes42>(table, data)
    }
}

impl Ntt126IfmaDFTExecute<Ntt126IfmaTableInv<Primes42>> for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_dft_execute(table: &Ntt126IfmaTableInv<Primes42>, data: &mut [u64]) {
        crate::ntt126_ifma::reference::ntt::intt126_ifma_ref::<Primes42>(table, data)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaFromZnx64 for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_from_znx64(res: &mut [u64], a: &[i64]) {
        unsafe { simd_b_from_znx64(a.len(), res, a) };
    }

    #[inline(always)]
    fn ntt126_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        unsafe { simd_b_from_znx64_masked(a.len(), res, a, mask) };
    }
}

impl Ntt126IfmaToZnx128 for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        unsafe { super::vec_znx_dft::simd_b_ntt126_ifma_to_znx128(divisor_is_n, res, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-specific addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaAdd for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl Ntt126IfmaAddAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_add_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_assign(res, a) };
    }
}

impl Ntt126IfmaSub for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl Ntt126IfmaSubAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_assign(res, a) };
    }
}

impl Ntt126IfmaSubNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_assign(res, a) };
    }
}

impl Ntt126IfmaNegate for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl Ntt126IfmaNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_negate_assign(res: &mut [u64]) {
        unsafe { simd_negate_assign(res) };
    }
}

impl Ntt126IfmaZero for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl Ntt126IfmaCopy for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaMulBbc for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_mul_bbc(meta: &Bbc126IfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        unsafe { vec_mat1col_product_bbc_ifma(meta, ell, res, ntt_coeff, prepared) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// b -> c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaCFromB for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // c format for IFMA = reduced residues: a[k] mod Q[k].
        // Values in [0, 2q) → cond_sub with q → [0, q).
        // res is typed as &mut [u32] for trait compatibility but is actually u64 data.
        let res_u64: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u64, res.len() / 2) };
        unsafe { simd_c_from_b(n, res_u64, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT120 Ntt* traits (DFT-domain arithmetic reuse)
//
// The ntt120 generic functions (ntt120_vec_znx_dft_add, etc.) require these
// traits. Since the 3-prime IFMA layout uses the same 4 x u64 per coefficient
// representation (with lane 3 as padding), the implementation is identical
// to the NTT120 version but uses Q_SHIFTED_NTT126IFMA for the 3 active lanes.
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT126Ifma {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl NttAddAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_assign(res, a) };
    }
}

impl NttSub for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl NttSubAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_assign(res, a) };
    }
}

impl NttSubNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_assign(res, a) };
    }
}

impl NttNegate for NTT126Ifma {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl NttNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        unsafe { simd_negate_assign(res) };
    }
}

impl NttZero for NTT126Ifma {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT126Ifma {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
