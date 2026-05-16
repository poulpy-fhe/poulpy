//! Raw AVX-512 kernels for `i128` arithmetic, shared by the NTT120 AVX-512 backends.
//!
//! This module contains the low-level SIMD kernels behind the `I128BigOps` and
//! `I128NormalizeOps` traits. The higher-level loop structure lives in the shared
//! defaults (`poulpy-cpu-ref`); this file provides the accelerated per-slice building blocks.
//!
//! # Memory layout
//!
//! Each `i128` is stored as `[lo: u64, hi: i64]` on little-endian x86-64.
//! Loading two consecutive `__m512i` reads `[lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]`
//! and `[lo4,hi4,lo5,hi5,lo6,hi6,lo7,hi7]`.
//! Deinterleave with `_mm512_permutex2var_epi64` using even/odd index tables:
//! `lo=[lo0,lo1,...,lo7]`, `hi=[hi0,hi1,...,hi7]`.
//!
//! # Correctness scope
//!
//! Normalization AVX-512 path: `base2k <= 64` (all `lsh` values).  `lsh == 0` and
//! `lsh != 0` have dedicated kernels; scalar fallback only when `base2k > 64`
//! or `n < 8`.
//!
//! [`I128NormalizeOps`]: poulpy_cpu_ref::reference::ntt120::I128NormalizeOps
//! [`I128BigOps`]: poulpy_cpu_ref::reference::ntt120::I128BigOps

use std::arch::x86_64::*;

use itertools::izip;
use poulpy_cpu_ref::reference::znx::{get_carry_i128, get_digit_i128};

// ──────────────────────────────────────────────────────────────────────────────
// Scalar fallback helpers (used as tails in AVX-512 kernels)
// ──────────────────────────────────────────────────────────────────────────────

#[inline(always)]
pub(super) fn nfc_middle_step_scalar(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    if lsh == 0 {
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

#[inline(always)]
pub(super) fn nfc_middle_step_add_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    if lsh == 0 {
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = r.wrapping_add(out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = r.wrapping_add(out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

#[inline(always)]
pub(super) fn nfc_middle_step_sub_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    if lsh == 0 {
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = r.wrapping_sub(out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = r.wrapping_sub(out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

#[inline(always)]
pub(super) fn nfc_middle_step_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            let digit = get_digit_i128(base2k, ri);
            let co = get_carry_i128(base2k, ri, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            let digit = get_digit_i128(base2k_lsh, ri);
            let co = get_carry_i128(base2k_lsh, ri, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

#[inline(always)]
pub(super) fn nfc_final_step_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            *r = get_digit_i128(base2k, get_digit_i128(base2k, ri) + *c) as i64;
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            *r = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, ri) << lsh) + *c) as i64;
        });
    }
}

#[inline(always)]
pub(super) fn nfc_final_step_add_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, get_digit_i128(base2k, *r as i128) + *c);
            *r = r.wrapping_add(out as i64);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, *r as i128) << lsh) + *c);
            *r = r.wrapping_add(out as i64);
        });
    }
}

#[inline(always)]
pub(super) fn nfc_final_step_sub_assign_scalar(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, get_digit_i128(base2k, *r as i128) + *c);
            *r = r.wrapping_sub(out as i64);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, *r as i128) << lsh) + *c);
            *r = r.wrapping_sub(out as i64);
        });
    }
}

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
// AVX-512 normalization kernels (base2k <= 64, any lsh)
// ──────────────────────────────────────────────────────────────────────────────

/// Precomputed shift constants shared across all three AVX-512 normalization kernels.
///
/// Fields with `__m128i` type are variable shift counts for `_mm512_srl/sll_epi64`.
/// Fields with `__m128i` type (sra) are shift counts for `_mm512_sra_epi64`.
struct NfcShifts512 {
    /// `_mm_cvtsi64_si128((64 - base2k_lsh) as i64)` -- left-shift for digit extraction.
    sll_b2klsh: __m128i,
    /// `_mm_cvtsi64_si128((64 - base2k_lsh) as i64)` -- arithmetic right-shift count for digit extraction.
    sra_b2klsh: __m128i,
    /// `_mm_cvtsi64_si128(base2k_lsh as i64)` -- right-shift for `co_lo`.
    srl_b2klsh: __m128i,
    /// `_mm_cvtsi64_si128(base2k_lsh as i64)` -- arithmetic right-shift count for `co_hi`.
    sra_b2klsh_val: __m128i,
    /// `_mm_cvtsi64_si128(lsh as i64)` -- left-shift for `digit << lsh`.
    sll_lsh: __m128i,
    /// `_mm_cvtsi64_si128((64 - base2k) as i64)` -- left-shift for out extraction.
    sll_b2k: __m128i,
    /// `_mm_cvtsi64_si128((64 - base2k) as i64)` -- arithmetic right-shift count for out extraction.
    sra_b2k: __m128i,
    /// `_mm_cvtsi64_si128(base2k as i64)` -- right-shift for `carry2_lo`.
    srl_b2k: __m128i,
    /// `_mm_cvtsi64_si128(base2k as i64)` -- arithmetic right-shift count for `carry2_hi`.
    sra_b2k_val: __m128i,
    /// Shift count 63 for sign extension.
    sra_63: __m128i,
}

impl NfcShifts512 {
    /// # Safety
    /// Must be called within a `#[target_feature(enable = "avx512f")]` context.
    #[inline(always)]
    unsafe fn new(base2k: u32, lsh: u32) -> Self {
        unsafe {
            let b2klsh = base2k - lsh;
            Self {
                sll_b2klsh: _mm_cvtsi64_si128((64 - b2klsh) as i64),
                sra_b2klsh: _mm_cvtsi64_si128((64 - b2klsh) as i64),
                srl_b2klsh: _mm_cvtsi64_si128(b2klsh as i64),
                sra_b2klsh_val: _mm_cvtsi64_si128(b2klsh as i64),
                sll_lsh: _mm_cvtsi64_si128(lsh as i64),
                sll_b2k: _mm_cvtsi64_si128((64 - base2k) as i64),
                sra_b2k: _mm_cvtsi64_si128((64 - base2k) as i64),
                srl_b2k: _mm_cvtsi64_si128(base2k as i64),
                sra_b2k_val: _mm_cvtsi64_si128(base2k as i64),
                sra_63: _mm_cvtsi64_si128(63),
            }
        }
    }
}

/// Shared inner loop body for `nfc_middle_step_avx512` and `nfc_middle_step_assign_avx512`.
///
/// Given deinterleaved split-i128 input `(lo_a, hi_a)` and carry `(lo_c, hi_c)`,
/// returns `(lo_out, new_lo_c, new_hi_c)`.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn nfc_middle_chunk_512(
    s: &NfcShifts512,
    lo_a: __m512i,
    hi_a: __m512i,
    lo_c: __m512i,
    hi_c: __m512i,
) -> (__m512i, __m512i, __m512i) {
    unsafe {
        // digit = get_digit_i128(base2k_lsh, a)
        let lo_dig = _mm512_sra_epi64(_mm512_sll_epi64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        let hi_dig = _mm512_sra_epi64(lo_dig, s.sra_63);

        // co = (a - digit) >> base2k_lsh
        let diff_lo = _mm512_sub_epi64(lo_a, lo_dig);
        // borrow: lo_dig >_u lo_a
        let borrow_mask: __mmask8 = _mm512_cmp_epu64_mask(lo_dig, lo_a, _MM_CMPINT_NLE);
        let borrow = _mm512_maskz_set1_epi64(borrow_mask, 1);
        let diff_hi = _mm512_sub_epi64(_mm512_sub_epi64(hi_a, hi_dig), borrow);
        let co_lo = _mm512_or_si512(
            _mm512_srl_epi64(diff_lo, s.srl_b2klsh),
            _mm512_sll_epi64(diff_hi, s.sll_b2klsh),
        );
        let co_hi = _mm512_sra_epi64(diff_hi, s.sra_b2klsh_val);

        // digit_shifted = digit << lsh
        let lo_dig_sh = _mm512_sll_epi64(lo_dig, s.sll_lsh);
        let hi_dig_sh = _mm512_sra_epi64(lo_dig_sh, s.sra_63);

        // d_plus_c = digit_shifted + carry
        let lo_dpc = _mm512_add_epi64(lo_dig_sh, lo_c);
        // carry: lo_dpc < lo_dig_sh (unsigned overflow)
        let carry1_mask: __mmask8 = _mm512_cmp_epu64_mask(lo_dpc, lo_dig_sh, _MM_CMPINT_LT);
        let carry1 = _mm512_maskz_set1_epi64(carry1_mask, 1);
        let hi_dpc = _mm512_add_epi64(_mm512_add_epi64(hi_dig_sh, hi_c), carry1);

        // out = get_digit_i128(base2k, d_plus_c)
        let lo_out = _mm512_sra_epi64(_mm512_sll_epi64(lo_dpc, s.sll_b2k), s.sra_b2k);
        let hi_out = _mm512_sra_epi64(lo_out, s.sra_63);

        // carry2 = (d_plus_c - out) >> base2k
        let diff2_lo = _mm512_sub_epi64(lo_dpc, lo_out);
        // borrow: lo_out >_u lo_dpc
        let borrow2_mask: __mmask8 = _mm512_cmp_epu64_mask(lo_out, lo_dpc, _MM_CMPINT_NLE);
        let borrow2 = _mm512_maskz_set1_epi64(borrow2_mask, 1);
        let diff2_hi = _mm512_sub_epi64(_mm512_sub_epi64(hi_dpc, hi_out), borrow2);
        let carry2_lo = _mm512_or_si512(_mm512_srl_epi64(diff2_lo, s.srl_b2k), _mm512_sll_epi64(diff2_hi, s.sll_b2k));
        let carry2_hi = _mm512_sra_epi64(diff2_hi, s.sra_b2k_val);

        // new_carry = co + carry2
        let new_lo_c = _mm512_add_epi64(co_lo, carry2_lo);
        // carry: new_lo_c < co_lo (unsigned overflow)
        let carry3_mask: __mmask8 = _mm512_cmp_epu64_mask(new_lo_c, co_lo, _MM_CMPINT_LT);
        let carry3 = _mm512_maskz_set1_epi64(carry3_mask, 1);
        let new_hi_c = _mm512_add_epi64(_mm512_add_epi64(co_hi, carry2_hi), carry3);

        (lo_out, new_lo_c, new_hi_c)
    }
}

/// Inner loop body for `nfc_final_step_assign_avx512`.
///
/// Given deinterleaved `lo_a` (sign-extended i64) and carry `lo_c` (low half only),
/// returns `lo_out`.
///
/// Note: `get_digit(base2k, d_plus_c)` with `base2k <= 64` depends only on the low
/// 64 bits of `d_plus_c`, so `hi_dpc` is never needed.
///
/// # Safety
/// Requires AVX-512F.
#[inline(always)]
unsafe fn nfc_final_chunk_512(s: &NfcShifts512, lo_a: __m512i, lo_c: __m512i) -> __m512i {
    unsafe {
        // digit = get_digit_i128(base2k_lsh, r) -- lo only since ri is i64.
        let lo_dig = _mm512_sra_epi64(_mm512_sll_epi64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        // d_plus_c lo = (digit << lsh) + carry_lo
        let lo_dpc = _mm512_add_epi64(_mm512_sll_epi64(lo_dig, s.sll_lsh), lo_c);
        // out = get_digit_i128(base2k, d_plus_c)  -- lo only since base2k <= 64
        _mm512_sra_epi64(_mm512_sll_epi64(lo_dpc, s.sll_b2k), s.sra_b2k)
    }
}

/// AVX-512 kernel for `nfc_middle_step` -- `i128` input + carry -> `i64` output.
///
/// Processes `n` elements with a scalar tail for `n % 8 != 0`.  Handles both
/// `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.  `base2k` must
/// be <= 64 (caller is responsible for this precondition).
///
/// # Safety
/// Requires AVX-512F.  `res`, `a`, `carry` must each have at least `n` elements.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_middle_step_avx512(base2k: u32, lsh: u32, n: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let a_ptr = a.as_ptr() as *const __m512i;
        let c_ptr = carry.as_mut_ptr() as *mut __m512i;
        let r_ptr = res.as_mut_ptr();

        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_deinterleave_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_interleave_lo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_interleave_hi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let a_lo8 = _mm512_loadu_si512(a_ptr.add(2 * i));
            let a_hi8 = _mm512_loadu_si512(a_ptr.add(2 * i + 1));
            let lo_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_lo, a_hi8);
            let hi_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_hi, a_hi8);

            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i) as *const __m512i);
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1) as *const __m512i);
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let hi_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_hi, c_hi8);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk_512(&s, lo_a, hi_a, lo_c, hi_c);

            // Store 8 i64 results directly (lo_out already in natural order)
            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_out);

            // Interleave carry back to memory layout
            let out_c_lo8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_lo, new_hi_c);
            let out_c_hi8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_hi, new_hi_c);
            _mm512_storeu_si512(c_ptr.add(2 * i), out_c_lo8);
            _mm512_storeu_si512(c_ptr.add(2 * i + 1), out_c_hi8);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_middle_step_scalar(
                base2k as usize,
                lsh as usize,
                &mut res[tail..],
                &a[tail..],
                &mut carry[tail..],
            );
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_middle_step_add_assign_avx512(
    base2k: u32,
    lsh: u32,
    n: usize,
    res: &mut [i64],
    a: &[i128],
    carry: &mut [i128],
) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let a_ptr = a.as_ptr() as *const __m512i;
        let c_ptr = carry.as_mut_ptr() as *mut __m512i;
        let r_ptr = res.as_mut_ptr();

        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_deinterleave_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_interleave_lo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_interleave_hi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let a_lo8 = _mm512_loadu_si512(a_ptr.add(2 * i));
            let a_hi8 = _mm512_loadu_si512(a_ptr.add(2 * i + 1));
            let lo_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_lo, a_hi8);
            let hi_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_hi, a_hi8);

            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i) as *const __m512i);
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1) as *const __m512i);
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let hi_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_hi, c_hi8);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk_512(&s, lo_a, hi_a, lo_c, hi_c);
            let lo_res = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);
            let lo_sum = _mm512_add_epi64(lo_res, lo_out);

            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_sum);

            let out_c_lo8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_lo, new_hi_c);
            let out_c_hi8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_hi, new_hi_c);
            _mm512_storeu_si512(c_ptr.add(2 * i), out_c_lo8);
            _mm512_storeu_si512(c_ptr.add(2 * i + 1), out_c_hi8);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_middle_step_add_assign_scalar(
                base2k as usize,
                lsh as usize,
                &mut res[tail..],
                &a[tail..],
                &mut carry[tail..],
            );
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_middle_step_sub_assign_avx512(
    base2k: u32,
    lsh: u32,
    n: usize,
    res: &mut [i64],
    a: &[i128],
    carry: &mut [i128],
) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let a_ptr = a.as_ptr() as *const __m512i;
        let c_ptr = carry.as_mut_ptr() as *mut __m512i;
        let r_ptr = res.as_mut_ptr();

        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_deinterleave_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_interleave_lo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_interleave_hi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let a_lo8 = _mm512_loadu_si512(a_ptr.add(2 * i));
            let a_hi8 = _mm512_loadu_si512(a_ptr.add(2 * i + 1));
            let lo_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_lo, a_hi8);
            let hi_a = _mm512_permutex2var_epi64(a_lo8, idx_deinterleave_hi, a_hi8);

            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i) as *const __m512i);
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1) as *const __m512i);
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let hi_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_hi, c_hi8);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk_512(&s, lo_a, hi_a, lo_c, hi_c);
            let lo_res = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);
            let lo_diff = _mm512_sub_epi64(lo_res, lo_out);

            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_diff);

            let out_c_lo8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_lo, new_hi_c);
            let out_c_hi8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_hi, new_hi_c);
            _mm512_storeu_si512(c_ptr.add(2 * i), out_c_lo8);
            _mm512_storeu_si512(c_ptr.add(2 * i + 1), out_c_hi8);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_middle_step_sub_assign_scalar(
                base2k as usize,
                lsh as usize,
                &mut res[tail..],
                &a[tail..],
                &mut carry[tail..],
            );
        }
    }
}

/// AVX-512 kernel for `nfc_middle_step_assign` -- in-place update of `i64` `res` with `i128` carry.
///
/// Like `nfc_middle_step_avx512` but the input `ai = *r as i128` is read from `res` itself.
/// Handles both `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_middle_step_assign_avx512(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let c_ptr = carry.as_mut_ptr() as *mut __m512i;
        let r_ptr = res.as_mut_ptr();

        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_deinterleave_hi = _mm512_loadu_si512(DEINTERLEAVE_HI.as_ptr() as *const __m512i);
        let idx_interleave_lo = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr() as *const __m512i);
        let idx_interleave_hi = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            // ai = *r as i128; sign-extend i64 to split i128 form.
            let lo_a = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);
            let hi_a = _mm512_srai_epi64::<63>(lo_a);

            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i) as *const __m512i);
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1) as *const __m512i);
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let hi_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_hi, c_hi8);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk_512(&s, lo_a, hi_a, lo_c, hi_c);

            // Store 8 i64 results directly
            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_out);

            // Interleave carry back to memory layout
            let out_c_lo8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_lo, new_hi_c);
            let out_c_hi8 = _mm512_permutex2var_epi64(new_lo_c, idx_interleave_hi, new_hi_c);
            _mm512_storeu_si512(c_ptr.add(2 * i), out_c_lo8);
            _mm512_storeu_si512(c_ptr.add(2 * i + 1), out_c_hi8);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_middle_step_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

/// AVX-512 kernel for `nfc_final_step_assign` -- flush `i128` carry into the last `i64` limb.
///
/// Computes `*r = get_digit(base2k, (get_digit(base2k_lsh, ri) << lsh) + carry)`.
/// Handles both `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_final_step_assign_avx512(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let c_ptr = carry.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr();

        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let lo_a = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);

            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i));
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1));
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);

            let lo_out = nfc_final_chunk_512(&s, lo_a, lo_c);
            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_out);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_final_step_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_final_step_add_assign_avx512(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let c_ptr = carry.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr();
        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let lo_res = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);
            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i));
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1));
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let lo_out = nfc_final_chunk_512(&s, lo_res, lo_c);
            let lo_sum = _mm512_add_epi64(lo_res, lo_out);
            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_sum);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_final_step_add_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nfc_final_step_sub_assign_avx512(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts512::new(base2k, lsh);
        let c_ptr = carry.as_ptr() as *const __m512i;
        let r_ptr = res.as_mut_ptr();
        let idx_deinterleave_lo = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const __m512i);

        let chunks = n / 8;
        for i in 0..chunks {
            let lo_res = _mm512_loadu_si512(r_ptr.add(8 * i) as *const __m512i);
            let c_lo8 = _mm512_loadu_si512(c_ptr.add(2 * i));
            let c_hi8 = _mm512_loadu_si512(c_ptr.add(2 * i + 1));
            let lo_c = _mm512_permutex2var_epi64(c_lo8, idx_deinterleave_lo, c_hi8);
            let lo_out = nfc_final_chunk_512(&s, lo_res, lo_c);
            let lo_diff = _mm512_sub_epi64(lo_res, lo_out);
            _mm512_storeu_si512(r_ptr.add(8 * i) as *mut __m512i, lo_diff);
        }

        let tail = chunks * 8;
        if tail < n {
            nfc_final_step_sub_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

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
    use super::{
        nfc_final_step_assign_avx512, nfc_final_step_assign_scalar, nfc_middle_step_assign_avx512, nfc_middle_step_assign_scalar,
        nfc_middle_step_avx512, nfc_middle_step_scalar, vi128_add_avx512, vi128_from_small_avx512, vi128_neg_from_small_avx512,
        vi128_negate_avx512, vi128_sub_avx512,
    };

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

    #[test]
    fn nfc_middle_step_avx512_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 0usize;
        let a = i128_data(n, 37i128);
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 3) % (1i128 << 20)).collect();

        let mut res_avx = vec![0i64; n];
        let mut carry_avx = carry_init.clone();
        let mut res_ref = vec![0i64; n];
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_middle_step_avx512(base2k as u32, lsh as u32, n, &mut res_avx, &a, &mut carry_avx) };
        nfc_middle_step_scalar(base2k, lsh, &mut res_ref, &a, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_middle_step res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_middle_step carry mismatch");
    }

    #[test]
    fn nfc_middle_step_assign_avx512_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 8usize;
        let init: Vec<i64> = (0..n).map(|i| (i as i64 * 5) % (1i64 << 20)).collect();
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 7) % (1i128 << 20)).collect();

        let mut res_avx = init.clone();
        let mut carry_avx = carry_init.clone();
        let mut res_ref = init.clone();
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_middle_step_assign_avx512(base2k as u32, lsh as u32, n, &mut res_avx, &mut carry_avx) };
        nfc_middle_step_assign_scalar(base2k, lsh, &mut res_ref, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_middle_step_assign res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_middle_step_assign carry mismatch");
    }

    #[test]
    fn nfc_final_step_assign_avx512_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 0usize;
        let init: Vec<i64> = (0..n).map(|i| (i as i64 * 3) % (1i64 << 20)).collect();
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 11) % (1i128 << 20)).collect();

        let mut res_avx = init.clone();
        let mut carry_avx = carry_init.clone();
        let mut res_ref = init.clone();
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_final_step_assign_avx512(base2k as u32, lsh as u32, n, &mut res_avx, &mut carry_avx) };
        nfc_final_step_assign_scalar(base2k, lsh, &mut res_ref, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_final_step_assign res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_final_step_assign carry mismatch");
    }
}
