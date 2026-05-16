// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code adapted from the AVX2 / FMA C kernels of the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The 256-bit AVX2 originals were widened to 512-bit AVX-512 and translated
// to Rust intrinsics; algorithmic structure is preserved one-to-one with the
// spqlios sources to keep semantics identical.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! Trait implementations for [`NTT120Avx512`](super::NTT120Avx512) — primitive NTT-domain operations.
//!
//! Implements all `Ntt*` traits from [`poulpy_cpu_ref::reference::ntt120`] for
//! [`NTT120Avx512`](super::NTT120Avx512).
//!
//! NTT forward/inverse execution uses the AVX-512F accelerated kernels from
//! [`super::ntt`]. BBC mat-vec products use the AVX-512F accelerated kernels
//! from [`super::mat_vec_avx512`]. Add/sub/negate on q120b elements use
//! AVX-512F lazy conditional subtraction (no division), pair-packing two
//! coefficients per `__m512i`. Domain conversion also uses AVX-512F kernels.

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_andnot_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256, _mm512_add_epi64, _mm512_broadcast_i64x4, _mm512_cmpgt_epi64_mask,
    _mm512_loadu_si512, _mm512_mask_sub_epi64, _mm512_set1_epi64, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_xor_si512,
};

use poulpy_cpu_ref::reference::ntt120::{
    NttAdd, NttAddAssign, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbb, NttMulBbc,
    NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateAssign, NttPackLeft1BlkX2, NttPackRight1BlkX2,
    NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, NttSub, NttSubAssign, NttSubNegateAssign, NttToZnx128, NttZero,
    mat_vec::{BbbMeta, BbcMeta},
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
    types::Q_SHIFTED,
};

use super::arithmetic_avx512::{
    b_from_znx64_avx512, b_from_znx64_masked_avx512, b_to_znx128_avx512, c_from_b_avx512, pack_left_1blk_x2_avx512,
    pack_right_1blk_x2_avx512, pairwise_pack_left_1blk_x2_avx512, pairwise_pack_right_1blk_x2_avx512,
    vec_mat1col_product_bbb_avx512,
};

use super::mat_vec_avx512::{
    vec_mat1col_product_bbc_avx512, vec_mat1col_product_x2_bbc_avx512, vec_mat2cols_product_x2_bbc_avx512,
};
use super::ntt::{intt_avx512, ntt_avx512};

use super::NTT120Avx512;

// ──────────────────────────────────────────────────────────────────────────────
// 256-bit lazy arithmetic helpers (used by the odd-coefficient tail)
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

/// 512-bit lazy reduction across two q120b coefficients (8 lanes = 2 × 4 primes).
///
/// `q_s` and `msb` are pre-broadcast across both 256-bit halves (per-prime constants
/// repeated). Mask-form `_mm512_cmpgt_epi64_mask` returns a `__mmask8` with bit `i` set
/// when `q_s_i > x_i (signed)`. We subtract `q_s` only where the mask bit is **clear**
/// (i.e., `x ≥ q_s` after the sign-flip), via `_mm512_mask_sub_epi64` with the inverted mask.
#[inline(always)]
unsafe fn lazy_reduce_512(x: __m512i, q_s: __m512i, msb: __m512i) -> __m512i {
    unsafe {
        let x_xor = _mm512_xor_si512(x, msb);
        let q_xor = _mm512_xor_si512(q_s, msb);
        let lt_mask = _mm512_cmpgt_epi64_mask(q_xor, x_xor);
        _mm512_mask_sub_epi64(x, !lt_mask, x, q_s)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AVX-512F lazy ops — process two q120b coefficients per __m512i iteration.
// Layout: [r0_A, r1_A, r2_A, r3_A,  r0_B, r1_B, r2_B, r3_B] (two 4-prime blocks).
// Per-prime constants (Q_SHIFTED, msb) are broadcast from the existing 4-u64
// arrays via `_mm512_broadcast_i64x4` / `_mm512_set1_epi64`.
// Scalar tail: when `n` is odd, a single trailing 256-bit iteration via the
// 4-lane `lazy_reduce`.
// ──────────────────────────────────────────────────────────────────────────────

/// Broadcast the 4 × u64 `Q_SHIFTED` constant into both halves of an `__m512i`.
#[inline(always)]
unsafe fn q_shifted_512() -> __m512i {
    unsafe { _mm512_broadcast_i64x4(_mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i)) }
}

/// `res[i] = lazy(a[i]) + lazy(b[i])` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_add_avx512(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut b_ptr = b.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            let bv = lazy_reduce_512(_mm512_loadu_si512(b_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_add_epi64(av, bv));
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            let bv = lazy_reduce(_mm256_loadu_si256(b_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(av, bv));
        }
    }
}

/// `res[i] = lazy(res[i]) + lazy(a[i])` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_add_assign_avx512(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let rv = lazy_reduce_512(_mm512_loadu_si512(r_ptr), q_s_512, msb_512);
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_add_epi64(rv, av));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(rv, av));
        }
    }
}

/// `res[i] = lazy(a[i]) + (q_s − lazy(b[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_sub_avx512(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut b_ptr = b.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            let bv = lazy_reduce_512(_mm512_loadu_si512(b_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_add_epi64(av, _mm512_sub_epi64(q_s_512, bv)));
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            let bv = lazy_reduce(_mm256_loadu_si256(b_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(av, _mm256_sub_epi64(q_s, bv)));
        }
    }
}

/// `res[i] = lazy(res[i]) + (q_s − lazy(a[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_sub_assign_avx512(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let rv = lazy_reduce_512(_mm512_loadu_si512(r_ptr), q_s_512, msb_512);
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_add_epi64(rv, _mm512_sub_epi64(q_s_512, av)));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(rv, _mm256_sub_epi64(q_s, av)));
        }
    }
}

/// `res[i] = lazy(a[i]) + (q_s − lazy(res[i]))` for `i ∈ 0..n` q120b elements.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_sub_negate_assign_avx512(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let rv = lazy_reduce_512(_mm512_loadu_si512(r_ptr), q_s_512, msb_512);
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_add_epi64(av, _mm512_sub_epi64(q_s_512, rv)));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(av, _mm256_sub_epi64(q_s, rv)));
        }
    }
}

/// `res[i] = q_s − lazy(a[i])` for `i ∈ 0..n` q120b elements.
///
/// **Output range:** For a zero input the result is `Q_SHIFTED[k]` (≡ 0 mod Q[k]), not `0`.
/// Output range is `(0, Q_SHIFTED[k]]`. Use `val % Q[k] == 0`, not `val == 0`, to test for zero.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_negate_avx512(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let av = lazy_reduce_512(_mm512_loadu_si512(a_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_sub_epi64(q_s_512, av));
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let av = lazy_reduce(_mm256_loadu_si256(a_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_sub_epi64(q_s, av));
        }
    }
}

/// `res[i] = q_s − lazy(res[i])` for `i ∈ 0..n` q120b elements.
///
/// **Output range:** For a zero input the result is `Q_SHIFTED[k]` (≡ 0 mod Q[k]), not `0`.
/// Output range is `(0, Q_SHIFTED[k]]`. Use `val % Q[k] == 0`, not `val == 0`, to test for zero.
#[target_feature(enable = "avx512f")]
unsafe fn ntt_negate_assign_avx512(n: usize, res: &mut [u64]) {
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let pairs = n / 2;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;
        for _ in 0..pairs {
            let rv = lazy_reduce_512(_mm512_loadu_si512(r_ptr), q_s_512, msb_512);
            _mm512_storeu_si512(r_ptr, _mm512_sub_epi64(q_s_512, rv));
            r_ptr = r_ptr.add(1);
        }
        if n & 1 != 0 {
            let q_s = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
            let msb = _mm256_set1_epi64x(i64::MIN);
            let rv = lazy_reduce(_mm256_loadu_si256(r_ptr as *const __m256i), q_s, msb);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_sub_epi64(q_s, rv));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT execution — AVX-512F butterfly
// ──────────────────────────────────────────────────────────────────────────────

impl NttDFTExecute<NttTable<Primes30>> for NTT120Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_avx512::<Primes30>(table, data) }
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT120Avx512 {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { intt_avx512::<Primes30>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttFromZnx64 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { b_from_znx64_avx512(a.len(), res, a) }
    }

    #[inline(always)]
    fn ntt_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { b_from_znx64_masked_avx512(a.len(), res, a, mask) }
    }
}

impl NttToZnx128 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { b_to_znx128_avx512(divisor_is_n, res, a) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Addition / subtraction / negation / copy / zero — AVX-512F lazy arithmetic
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT120Avx512 {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_add_avx512(res.len() / 4, res, a, b) }
    }
}

impl NttAddAssign for NTT120Avx512 {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_add_assign_avx512(res.len() / 4, res, a) }
    }
}

impl NttSub for NTT120Avx512 {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_sub_avx512(res.len() / 4, res, a, b) }
    }
}

impl NttSubAssign for NTT120Avx512 {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_sub_assign_avx512(res.len() / 4, res, a) }
    }
}

impl NttSubNegateAssign for NTT120Avx512 {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_sub_negate_assign_avx512(res.len() / 4, res, a) }
    }
}

impl NttNegate for NTT120Avx512 {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_negate_avx512(res.len() / 4, res, a) }
    }
}

impl NttNegateAssign for NTT120Avx512 {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { ntt_negate_assign_avx512(res.len() / 4, res) }
    }
}

impl NttZero for NTT120Avx512 {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT120Avx512 {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbb for NTT120Avx512 {
    #[inline(always)]
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { vec_mat1col_product_bbb_avx512(meta, ell, res, a, b) }
    }
}

impl NttMulBbc for NTT120Avx512 {
    #[inline(always)]
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { vec_mat1col_product_bbc_avx512(meta, ell, res, ntt_coeff, prepared) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// q120b → q120c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttCFromB for NTT120Avx512 {
    #[inline(always)]
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { c_from_b_avx512(n, res, a) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbc1ColX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { vec_mat1col_product_x2_bbc_avx512::<false>(meta, ell, res, a, b) }
    }
}

impl NttMulBbc2ColsX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { vec_mat2cols_product_x2_bbc_avx512(meta, ell, res, a, b) }
    }
}

impl NttExtract1BlkContiguous for NTT120Avx512 {
    #[inline(always)]
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { crate::ntt120_avx512::vmp::extract_1blk_from_contiguous_q120b_avx512(n, row_max, blk, dst, src) }
    }
}

impl NttPackLeft1BlkX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { pack_left_1blk_x2_avx512(dst, a, row_count, row_stride, blk) }
    }
}

impl NttPackRight1BlkX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { pack_right_1blk_x2_avx512(dst, a, row_count, row_stride, blk) }
    }
}

impl NttPairwisePackLeft1BlkX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { pairwise_pack_left_1blk_x2_avx512(dst, a, b, row_count, row_stride, blk) }
    }
}

impl NttPairwisePackRight1BlkX2 for NTT120Avx512 {
    #[inline(always)]
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], row_count: usize, row_stride: usize, blk: usize) {
        // SAFETY: NTT120Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { pairwise_pack_right_1blk_x2_avx512(dst, a, b, row_count, row_stride, blk) }
    }
}
