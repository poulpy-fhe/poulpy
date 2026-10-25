//! Raw AVX-512 kernels for `i128` arithmetic, shared by the NTT4x30 AVX-512 backends.
//!
//! This module contains the low-level SIMD kernels behind the `I128BigOps` trait.
//! The higher-level loop structure lives in the shared defaults (`poulpy-cpu-portable`);
//! this file provides the accelerated per-slice building blocks.
//!
//! # Memory layout
//!
//! Each `i128` is stored as `[lo: u64, hi: i64]` on little-endian x86-64.
//! Loading two consecutive `__m512i` reads `[lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]`
//! and `[lo4,hi4,lo5,hi5,lo6,hi6,lo7,hi7]`.
//! Deinterleave with `_mm512_permutex2var_epi64` using even/odd index tables:
//! `lo=[lo0,lo1,...,lo7]`, `hi=[hi0,hi1,...,hi7]`.
//!
//! [`I128BigOps`]: poulpy_cpu_portable::kernels::ntt4x30::I128BigOps

use std::arch::x86_64::*;

// ──────────────────────────────────────────────────────────────────────────────
// AVX-512 helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Index tables for deinterleaving / interleaving 8 x i128 stored as
/// `[lo0,hi0,lo1,hi1,...,lo7,hi7]` across two `__m512i` registers.
const DEINTERLEAVE_LO: [i64; 8] = [0, 2, 4, 6, 8, 10, 12, 14]; // even indices
const DEINTERLEAVE_HI: [i64; 8] = [1, 3, 5, 7, 9, 11, 13, 15]; // odd indices
const INTERLEAVE_LO: [i64; 8] = [0, 8, 1, 9, 2, 10, 3, 11]; // first 4 pairs
const INTERLEAVE_HI: [i64; 8] = [4, 12, 5, 13, 6, 14, 7, 15]; // second 4 pairs

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice i128 arithmetic -- Phase 2 VecZnxBig AVX-512
//
// All kernels process 8 x i128 elements per chunk in deinterleaved split
// (lo, hi) form.  A scalar tail handles n % 8 != 0.
//
// Layout: each i128 is stored as [lo: u64, hi: i64] (little-endian x86-64).
// Loading two consecutive __m512i gives [lo0,hi0,...,lo3,hi3] and [lo4,hi4,...,lo7,hi7].
// Deinterleave with _mm512_permutex2var_epi64 using even/odd index tables.
// After computation, interleave back with the inverse permutation before storing.
// ──────────────────────────────────────────────────────────────────────────────

/// Load 8 i128s from `a_ptr[2*i .. 2*i+2]` into deinterleaved split form `(lo, hi)`.
///
/// # Safety
/// Requires AVX-512F.  `a_ptr` must be valid for at least `2*(i+1)` x `__m512i` loads.
#[inline(always)]
unsafe fn load8_i128(a_ptr: *const __m512i, i: usize, idx_lo: __m512i, idx_hi: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let a_lo8 = _mm512_loadu_si512(a_ptr.add(2 * i));
        let a_hi8 = _mm512_loadu_si512(a_ptr.add(2 * i + 1));
        let lo = _mm512_permutex2var_epi64(a_lo8, idx_lo, a_hi8);
        let hi = _mm512_permutex2var_epi64(a_lo8, idx_hi, a_hi8);
        (lo, hi)
    }
}

/// Load 8 i64s from `a_ptr[i]` (one `__m512i`) and sign-extend to split i128 form `(lo, hi)`.
///
/// # Safety
/// Requires AVX-512F.  `a_ptr` must be valid for at least `i+1` x `__m512i` loads.
#[inline(always)]
unsafe fn load8_i64_as_i128(a_ptr: *const __m512i, i: usize) -> (__m512i, __m512i) {
    unsafe {
        let lo = _mm512_loadu_si512(a_ptr.add(i));
        let hi = _mm512_srai_epi64::<63>(lo); // sign extension (native)
        (lo, hi)
    }
}

/// Store 8 i128s in deinterleaved split form `(lo_r, hi_r)` to `r_ptr[2*i .. 2*i+2]`.
///
/// # Safety
/// Requires AVX-512F.  `r_ptr` must be valid for at least `2*(i+1)` x `__m512i` stores.
#[inline(always)]
unsafe fn store8_i128(r_ptr: *mut __m512i, i: usize, lo_r: __m512i, hi_r: __m512i, idx_lo: __m512i, idx_hi: __m512i) {
    unsafe {
        let out_lo8 = _mm512_permutex2var_epi64(lo_r, idx_lo, hi_r);
        let out_hi8 = _mm512_permutex2var_epi64(lo_r, idx_hi, hi_r);
        _mm512_storeu_si512(r_ptr.add(2 * i), out_lo8);
        _mm512_storeu_si512(r_ptr.add(2 * i + 1), out_hi8);
    }
}

/// Signed 64x64 -> 128-bit multiplication in split form.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn mul8_i64_to_i128(lo_a: __m512i, lo_b: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let mask32 = _mm512_set1_epi64(0xffff_ffff);

        let a_hi32 = _mm512_srli_epi64(lo_a, 32);
        let b_hi32 = _mm512_srli_epi64(lo_b, 32);

        let p0 = _mm512_mul_epu32(lo_a, lo_b);
        let p1 = _mm512_mul_epu32(lo_a, b_hi32);
        let p2 = _mm512_mul_epu32(a_hi32, lo_b);
        let p3 = _mm512_mul_epu32(a_hi32, b_hi32);

        let mid_low = _mm512_add_epi64(
            _mm512_add_epi64(_mm512_srli_epi64(p0, 32), _mm512_and_si512(p1, mask32)),
            _mm512_and_si512(p2, mask32),
        );
        let lo = _mm512_or_si512(_mm512_and_si512(p0, mask32), _mm512_slli_epi64(mid_low, 32));

        let hi_unsigned = _mm512_add_epi64(
            _mm512_add_epi64(p3, _mm512_srli_epi64(p1, 32)),
            _mm512_add_epi64(_mm512_srli_epi64(p2, 32), _mm512_srli_epi64(mid_low, 32)),
        );

        let zero = _mm512_setzero_si512();
        let sign_a = _mm512_maskz_set1_epi64(_mm512_cmp_epi64_mask(lo_a, zero, _MM_CMPINT_LT), -1);
        let sign_b = _mm512_maskz_set1_epi64(_mm512_cmp_epi64_mask(lo_b, zero, _MM_CMPINT_LT), -1);
        let hi = _mm512_sub_epi64(
            _mm512_sub_epi64(hi_unsigned, _mm512_and_si512(sign_a, lo_b)),
            _mm512_and_si512(sign_b, lo_a),
        );

        (lo, hi)
    }
}

/// `res[i] = (a[i] as i128).wrapping_mul(b[i] as i128)` for `n` elements.
///
/// # Safety
/// Requires AVX-512F. All slices must have at least `n` elements and `res` must
/// not alias `a` or `b`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn vi128_hadamard_i64_avx512(n: usize, res: &mut [i128], a: &[i64], b: &[i64]) {
    assert!(res.len() >= n);
    assert!(a.len() >= n);
    assert!(b.len() >= n);

    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;

        for i in 0..chunks {
            let (lo_a, _) = load8_i64_as_i128(a_ptr, i);
            let (lo_b, _) = load8_i64_as_i128(b_ptr, i);
            let (lo_r, hi_r) = mul8_i64_to_i128(lo_a, lo_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }

        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = (ai as i128).wrapping_mul(bi as i128));
    }
}

/// 128-bit addition in split form: `(lo_r, hi_r) = (lo_a + lo_b, hi_a + hi_b + carry)`.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn add8_i128(lo_a: __m512i, hi_a: __m512i, lo_b: __m512i, hi_b: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let lo_r = _mm512_add_epi64(lo_a, lo_b);
        // carry: lo_r < lo_a (unsigned) means addition overflowed
        let carry_mask: __mmask8 = _mm512_cmp_epu64_mask(lo_r, lo_a, _MM_CMPINT_LT);
        let carry_one = _mm512_maskz_set1_epi64(carry_mask, 1);
        let hi_r = _mm512_add_epi64(_mm512_add_epi64(hi_a, hi_b), carry_one);
        (lo_r, hi_r)
    }
}

/// 128-bit subtraction in split form: `(lo_r, hi_r) = (lo_a - lo_b, hi_a - hi_b - borrow)`.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn sub8_i128(lo_a: __m512i, hi_a: __m512i, lo_b: __m512i, hi_b: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let lo_r = _mm512_sub_epi64(lo_a, lo_b);
        // borrow: lo_b > lo_a (unsigned) means subtraction underflowed
        let borrow_mask: __mmask8 = _mm512_cmp_epu64_mask(lo_b, lo_a, _MM_CMPINT_NLE);
        let borrow_one = _mm512_maskz_set1_epi64(borrow_mask, 1);
        let hi_r = _mm512_sub_epi64(_mm512_sub_epi64(hi_a, hi_b), borrow_one);
        (lo_r, hi_r)
    }
}

/// 128-bit negation in split form: `(lo_r, hi_r) = -(lo_a, hi_a)`.
///
/// Uses the identity `-(lo, hi) = (0 - lo, ~hi + carry)` where `carry = (lo == 0) ? 1 : 0`.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn neg8_i128(lo_a: __m512i, hi_a: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let zero = _mm512_setzero_si512();
        let lo_r = _mm512_sub_epi64(zero, lo_a);
        // carry into hi = 1 iff lo_a == 0  (two's-complement negation carry)
        let eq_mask: __mmask8 = _mm512_cmpeq_epi64_mask(lo_a, zero);
        let carry_one = _mm512_maskz_set1_epi64(eq_mask, 1);
        // hi_r = ~hi_a + carry  =  xor(hi_a, all_ones) + carry_one
        let not_hi = _mm512_xor_si512(hi_a, _mm512_set1_epi64(-1));
        let hi_r = _mm512_add_epi64(not_hi, carry_one);
        (lo_r, hi_r)
    }
}

/// `res[i] = a[i].wrapping_add(b[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_add_avx512(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_b, hi_b) = load8_i128(b_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = add8_i128(lo_a, hi_a, lo_b, hi_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_add(bi));
    }
}

/// `res[i] = res[i].wrapping_add(a[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_add_assign_avx512(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = add8_i128(lo_r, hi_r, lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai));
    }
}

/// `res[i] = a[i].wrapping_add(b[i] as i128)` for `n` elements (`b` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_add_small_avx512(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_b, hi_b) = load8_i64_as_i128(b_ptr, i);
            let (lo_r, hi_r) = add8_i128(lo_a, hi_a, lo_b, hi_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_add(bi as i128));
    }
}

/// `res[i] = res[i].wrapping_add(a[i] as i128)` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_add_small_assign_avx512(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = add8_i128(lo_r, hi_r, lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai as i128));
    }
}

/// `res[i] = a[i].wrapping_sub(b[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_avx512(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_b, hi_b) = load8_i128(b_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = sub8_i128(lo_a, hi_a, lo_b, hi_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_sub(bi));
    }
}

/// `res[i] = res[i].wrapping_sub(a[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_assign_avx512(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = sub8_i128(lo_r, hi_r, lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai));
    }
}

/// `res[i] = a[i].wrapping_sub(res[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_negate_assign_avx512(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = sub8_i128(lo_a, hi_a, lo_r, hi_r); // a - res
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = ai.wrapping_sub(*r));
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(b[i])` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_small_a_avx512(n: usize, res: &mut [i128], a: &[i64], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            let (lo_b, hi_b) = load8_i128(b_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = sub8_i128(lo_a, hi_a, lo_b, hi_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = (ai as i128).wrapping_sub(bi));
    }
}

/// `res[i] = a[i].wrapping_sub(b[i] as i128)` for `n` elements (`b` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_small_b_avx512(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let b_ptr = b.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_b, hi_b) = load8_i64_as_i128(b_ptr, i);
            let (lo_r, hi_r) = sub8_i128(lo_a, hi_a, lo_b, hi_b);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .zip(b[tail..n].iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_sub(bi as i128));
    }
}

/// `res[i] = res[i].wrapping_sub(a[i] as i128)` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_small_assign_avx512(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = sub8_i128(lo_r, hi_r, lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai as i128));
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(res[i])` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_sub_small_negate_assign_avx512(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = sub8_i128(lo_a, hi_a, lo_r, hi_r); // a - res
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = (ai as i128).wrapping_sub(*r));
    }
}

/// `res[i] = a[i].wrapping_neg()` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_negate_avx512(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i128(a_ptr, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = neg8_i128(lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = ai.wrapping_neg());
    }
}

/// `res[i] = res[i].wrapping_neg()` for `n` i128 elements.
///
/// # Safety
/// Requires AVX-512F.  Slice must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_negate_assign_avx512(n: usize, res: &mut [i128]) {
    unsafe {
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_r, hi_r) = load8_i128(r_ptr as *const __m512i, i, idx_lo, idx_hi);
            let (lo_r, hi_r) = neg8_i128(lo_r, hi_r);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n].iter_mut().for_each(|r| *r = r.wrapping_neg());
    }
}

/// `res[i] = a[i] as i128` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_from_small_avx512(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            store8_i128(r_ptr, i, lo_a, hi_a, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        for j in tail..n {
            res[j] = a[j] as i128;
        }
    }
}

/// `res[i] = -(a[i] as i128)` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX-512F.  All slices must have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn vi128_neg_from_small_avx512(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr() as *mut __m512i;
        let idx_ilo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_ihi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);
        let chunks = n / 8;
        for i in 0..chunks {
            let (lo_a, hi_a) = load8_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = neg8_i128(lo_a, hi_a);
            store8_i128(r_ptr, i, lo_r, hi_r, idx_ilo, idx_ihi);
        }
        let tail = chunks * 8;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = -(ai as i128));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::{vi128_add_avx512, vi128_from_small_avx512, vi128_neg_from_small_avx512, vi128_negate_avx512, vi128_sub_avx512};

    fn i128_data(n: usize, seed: i128) -> Vec<i128> {
        (0..n).map(|i| (i as i128 * seed + seed / 3) % (1i128 << 80)).collect()
    }

    fn i64_data(n: usize, seed: i64) -> Vec<i64> {
        (0..n).map(|i| i as i64 * seed - seed / 2).collect()
    }

    #[test]
    fn vi128_add_avx512_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x1_0000_0001i128);
        let b = i128_data(n, 0x0_FFFF_FFFFi128);
        let expected: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_add_avx512(n, &mut res, &a, &b) };
        assert_eq!(res, expected, "vi128_add_avx512 mismatch");
    }

    #[test]
    fn vi128_sub_avx512_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x2_0000_0003i128);
        let b = i128_data(n, 0x1_0000_0001i128);
        let expected: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_sub_avx512(n, &mut res, &a, &b) };
        assert_eq!(res, expected, "vi128_sub_avx512 mismatch");
    }

    #[test]
    fn vi128_negate_avx512_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x1_2345_6789i128);
        let expected: Vec<i128> = a.iter().map(|x| -x).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_negate_avx512(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_negate_avx512 mismatch");
    }

    #[test]
    fn vi128_from_small_avx512_vs_scalar() {
        let n = 64usize;
        let a = i64_data(n, 12345);
        let expected: Vec<i128> = a.iter().map(|&x| x as i128).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_from_small_avx512(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_from_small_avx512 mismatch");
    }

    #[test]
    fn vi128_neg_from_small_avx512_vs_scalar() {
        let n = 64usize;
        let a = i64_data(n, 99);
        let expected: Vec<i128> = a.iter().map(|&x| -(x as i128)).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_neg_from_small_avx512(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_neg_from_small_avx512 mismatch");
    }
}
