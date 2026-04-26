// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! Trait implementations for [`NTT120Avx`](super::NTT120Avx) — primitive NTT-domain operations.
//!
//! Implements all `Ntt*` traits from [`poulpy_cpu_ref::reference::ntt120`] for
//! [`NTT120Avx`](super::NTT120Avx).
//!
//! NTT forward/inverse execution uses the AVX2-accelerated kernels from
//! [`super::ntt`]. BBC mat-vec products use the AVX2-accelerated kernels
//! from [`super::mat_vec_avx`]. Add/sub/negate on q120b elements use AVX2
//! lazy conditional subtraction (no division). Domain conversion also uses
//! AVX2 kernels.

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_andnot_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use poulpy_cpu_ref::reference::ntt120::{
    NttAdd, NttAddAssign, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbb, NttMulBbc,
    NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateAssign, NttPackLeft1BlkX2, NttPackRight1BlkX2,
    NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, NttSub, NttSubAssign, NttSubNegateAssign, NttToZnx128, NttZero,
    mat_vec::{BbbMeta, BbcMeta, extract_1blk_from_contiguous_q120b_ref},
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
    types::Q_SHIFTED,
};

use super::arithmetic_avx::{
    b_from_znx64_avx2, b_from_znx64_masked_avx2, b_to_znx128_avx2, c_from_b_avx2, pack_left_1blk_x2_avx2,
    pack_right_1blk_x2_avx2, pairwise_pack_left_1blk_x2_avx2, pairwise_pack_right_1blk_x2_avx2, vec_mat1col_product_bbb_avx2,
};

use super::mat_vec_avx::{vec_mat1col_product_bbc_avx2, vec_mat1col_product_x2_bbc_avx2, vec_mat2cols_product_x2_bbc_avx2};
use super::ntt::{intt_avx2, ntt_avx2};

use super::NTT120Avx;

// ──────────────────────────────────────────────────────────────────────────────
// AVX2 lazy arithmetic helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Lazy reduction: bring each 64-bit lane of `x ∈ [0, 2·q_s)` into `[0, q_s)`.
///
/// Subtracts `q_s` from each lane where `x ≥ q_s` (unsigned), using the
/// sign-flip trick: `a ≥ᵤ b  ⟺  (a ⊕ msb) ≥ₛ (b ⊕ msb)`.
///
/// Valid for Primes30 because `q_s = Q[k] << 33 < 2^63` for all four primes,
/// so XOR with the MSB maps both operands into a well-ordered signed range.
#[inline(always)]
unsafe fn lazy_reduce(x: __m256i, q_s: __m256i, msb: __m256i) -> __m256i {
    unsafe {
        let x_xor = _mm256_xor_si256(x, msb);
        let q_xor = _mm256_xor_si256(q_s, msb);
        // cmpgt(q_xor, x_xor) gives all-ones when q_s >_u x, i.e. x <_u q_s (no subtract needed).
        let lt = _mm256_cmpgt_epi64(q_xor, x_xor);
        _mm256_sub_epi64(x, _mm256_andnot_si256(lt, q_s))
    }
}

/// `res[i] = lazy(a[i]) + lazy(b[i])` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx2")]
unsafe fn ntt_add_avx2(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut b_ptr = b.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            let bv = lazy_reduce(_mm256_loadu_si256(b_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(av, bv));
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = lazy(res[i]) + lazy(a[i])` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx2")]
unsafe fn ntt_add_assign_avx2(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(rv, av));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = lazy(a[i]) + (q_s − lazy(b[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx2")]
unsafe fn ntt_sub_avx2(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut b_ptr = b.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            let bv = lazy_reduce(_mm256_loadu_si256(b_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(av, _mm256_sub_epi64(q_s, bv)));
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = lazy(res[i]) + (q_s − lazy(a[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx2")]
unsafe fn ntt_sub_assign_avx2(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(rv, _mm256_sub_epi64(q_s, av)));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = lazy(a[i]) + (q_s − lazy(res[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx2")]
unsafe fn ntt_sub_negate_assign_avx2(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(av, _mm256_sub_epi64(q_s, rv)));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = q_s − lazy(a[i])` for `i ∈ 0..n` q120b elements.
///
/// **Output range:** For a zero input the result is `Q_SHIFTED[k]` (≡ 0 mod Q[k]), not `0`.
/// Output range is `(0, Q_SHIFTED[k]]`. Use `val % Q[k] == 0`, not `val == 0`, to test for zero.
#[target_feature(enable = "avx2")]
unsafe fn ntt_negate_avx2(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_sub_epi64(q_s, av));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// `res[i] = q_s − lazy(res[i])` for `i ∈ 0..n` q120b elements.
///
/// **Output range:** For a zero input the result is `Q_SHIFTED[k]` (≡ 0 mod Q[k]), not `0`.
/// Output range is `(0, Q_SHIFTED[k]]`. Use `val % Q[k] == 0`, not `val == 0`, to test for zero.
#[target_feature(enable = "avx2")]
unsafe fn ntt_negate_assign_avx2(n: usize, res: &mut [u64]) {
    unsafe {
        let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let msb = _mm256_set1_epi64x(i64::MIN);
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
        for _ in 0..n {
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr, _mm256_sub_epi64(q_s, rv));
            r_ptr = r_ptr.add(1);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT execution — AVX2 butterfly
// ──────────────────────────────────────────────────────────────────────────────

impl NttDFTExecute<NttTable<Primes30>> for NTT120Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_avx2::<Primes30>(table, data) }
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT120Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { intt_avx2::<Primes30>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttFromZnx64 for NTT120Avx {
    #[inline(always)]
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { b_from_znx64_avx2(a.len(), res, a) }
    }

    #[inline(always)]
    fn ntt_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { b_from_znx64_masked_avx2(a.len(), res, a, mask) }
    }
}

impl NttToZnx128 for NTT120Avx {
    #[inline(always)]
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { b_to_znx128_avx2(divisor_is_n, res, a) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Addition / subtraction / negation / copy / zero — AVX2 lazy arithmetic
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT120Avx {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_add_avx2(res.len() / 4, res, a, b) }
    }
}

impl NttAddAssign for NTT120Avx {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_add_assign_avx2(res.len() / 4, res, a) }
    }
}

impl NttSub for NTT120Avx {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_sub_avx2(res.len() / 4, res, a, b) }
    }
}

impl NttSubAssign for NTT120Avx {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_sub_assign_avx2(res.len() / 4, res, a) }
    }
}

impl NttSubNegateAssign for NTT120Avx {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_sub_negate_assign_avx2(res.len() / 4, res, a) }
    }
}

impl NttNegate for NTT120Avx {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_negate_avx2(res.len() / 4, res, a) }
    }
}

impl NttNegateAssign for NTT120Avx {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_negate_assign_avx2(res.len() / 4, res) }
    }
}

impl NttZero for NTT120Avx {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT120Avx {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbb for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat1col_product_bbb_avx2(meta, ell, res, a, b) }
    }
}

impl NttMulBbc for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat1col_product_bbc_avx2(meta, ell, res, ntt_coeff, prepared) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// q120b → q120c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttCFromB for NTT120Avx {
    #[inline(always)]
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { c_from_b_avx2(n, res, a) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbc1ColX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat1col_product_x2_bbc_avx2(meta, ell, res, a, b) }
    }
}

impl NttMulBbc2ColsX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat2cols_product_x2_bbc_avx2(meta, ell, res, a, b) }
    }
}

impl NttExtract1BlkContiguous for NTT120Avx {
    #[inline(always)]
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, dst, src);
    }
}

impl NttPackLeft1BlkX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { pack_left_1blk_x2_avx2(dst, a, row_count, row_stride, blk) }
    }
}

impl NttPackRight1BlkX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { pack_right_1blk_x2_avx2(dst, a, row_count, row_stride, blk) }
    }
}

impl NttPairwisePackLeft1BlkX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { pairwise_pack_left_1blk_x2_avx2(dst, a, b, row_count, row_stride, blk) }
    }
}

impl NttPairwisePackRight1BlkX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { pairwise_pack_right_1blk_x2_avx2(dst, a, b, row_count, row_stride, blk) }
    }
}
