//! AVX2-accelerated kernels for VecZnxBig operations (NTT120 backend).
//!
//! Provides:
//! - **Normalization hot-path kernels** (`nfc_middle_step_avx2`, etc.): AVX2
//!   implementations of the three [`I128NormalizeOps`] trait methods.  The outer
//!   loop logic lives in `poulpy-hal`; backends dispatch through the trait.
//! - **Per-slice i128 arithmetic kernels** (`vi128_*_avx2`): AVX2 implementations
//!   of all [`I128BigOps`] trait methods for add/sub/negate/from_small on `i128` slices.
//!
//! All kernels process 4 × i128 elements per chunk in deinterleaved split
//! (lo, hi) form, with a scalar tail for `n % 4 != 0`.
//!
//! # Memory layout
//!
//! Each `i128` is stored as `[lo: u64, hi: i64]` on little-endian x86-64.
//! Loading two consecutive `__m256i` reads `[lo0,hi0,lo1,hi1]` and `[lo2,hi2,lo3,hi3]`.
//! Deinterleave with `unpacklo/hi_epi64`: `lo=[lo0,lo2,lo1,lo3]`, `hi=[hi0,hi2,hi1,hi3]`.
//!
//! # Correctness scope
//!
//! Normalization AVX2 path: `base2k ≤ 64` (all `lsh` values).  `lsh == 0` and
//! `lsh != 0` have dedicated kernels; scalar fallback only when `base2k > 64`
//! or `n < 4`.
//!
//! [`I128NormalizeOps`]: poulpy_cpu_ref::reference::ntt120::I128NormalizeOps
//! [`I128BigOps`]: poulpy_cpu_ref::reference::ntt120::I128BigOps

use std::arch::x86_64::*;

use itertools::izip;
use poulpy_cpu_ref::reference::znx::{get_carry_i128, get_digit_i128};

// ──────────────────────────────────────────────────────────────────────────────
// Scalar fallback helpers (used as tails in AVX2 kernels)
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

// ──────────────────────────────────────────────────────────────────────────────
// AVX2 helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Arithmetic right shift of all four i64 lanes by `imm` bits (0 ≤ imm ≤ 64).
///
/// Emulates `_mm256_srai_epi64` (unavailable in AVX2) by broadcasting the sign
/// bit with `srai_epi32` and masking the upper bits after a logical shift.
///
/// # Safety
/// Must be called within an `#[target_feature(enable = "avx2")]` context.
#[inline(always)]
unsafe fn sra_epi64(v: __m256i, imm: u32) -> __m256i {
    debug_assert!(imm <= 64, "sra_epi64: imm={imm} out of range [0, 64]");
    unsafe {
        // Broadcast the sign bit of each i64 lane into all 32-bit slots.
        // shuffle_epi32(v, 0xF5) copies the high 32-bit half of each 64-bit lane to both halves.
        let sign = _mm256_srai_epi32(_mm256_shuffle_epi32(v, 0xF5), 31);
        // Logical right shift.
        let shifted = _mm256_srl_epi64(v, _mm_cvtsi64_si128(imm as i64));
        // Mask for the upper `imm` bits (bits that should be filled with the sign).
        // all_ones << (64 - imm) is 0 when imm=0, all_ones when imm=64.
        let all_ones = _mm256_cmpeq_epi64(v, v);
        let mask = _mm256_sll_epi64(all_ones, _mm_cvtsi64_si128((64 - imm) as i64));
        _mm256_or_si256(shifted, _mm256_and_si256(sign, mask))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AVX2 normalization kernels (base2k <= 64, any lsh)
// ──────────────────────────────────────────────────────────────────────────────

/// Precomputed shift constants shared across all three AVX2 normalization kernels.
///
/// Fields with `__m128i` type are variable shift counts for `_mm256_srl/sll_epi64`.
/// Fields with `u32` type are immediates for `sra_epi64`.
struct NfcShifts {
    /// `_mm_cvtsi64_si128((64 - base2k_lsh) as i64)` — left-shift for digit extraction.
    sll_b2klsh: __m128i,
    /// `64 - base2k_lsh` — arithmetic right-shift immediate for digit extraction.
    sra_b2klsh: u32,
    /// `_mm_cvtsi64_si128(base2k_lsh as i64)` — right-shift for `co_lo`.
    srl_b2klsh: __m128i,
    /// `base2k_lsh` — arithmetic right-shift immediate for `co_hi`.
    b2klsh: u32,
    /// `_mm_cvtsi64_si128(lsh as i64)` — left-shift for `digit << lsh`.
    sll_lsh: __m128i,
    /// `_mm_cvtsi64_si128((64 - base2k) as i64)` — left-shift for out extraction.
    sll_b2k: __m128i,
    /// `64 - base2k` — arithmetic right-shift immediate for out extraction.
    sra_b2k: u32,
    /// `_mm_cvtsi64_si128(base2k as i64)` — right-shift for `carry2_lo`.
    srl_b2k: __m128i,
    /// `base2k` — arithmetic right-shift immediate for `carry2_hi`.
    b2k: u32,
    msb: __m256i,
    zero: __m256i,
}

impl NfcShifts {
    /// # Safety
    /// Must be called within a `#[target_feature(enable = "avx2")]` context.
    #[inline(always)]
    unsafe fn new(base2k: u32, lsh: u32) -> Self {
        unsafe {
            let b2klsh = base2k - lsh;
            Self {
                sll_b2klsh: _mm_cvtsi64_si128((64 - b2klsh) as i64),
                sra_b2klsh: 64 - b2klsh,
                srl_b2klsh: _mm_cvtsi64_si128(b2klsh as i64),
                b2klsh,
                sll_lsh: _mm_cvtsi64_si128(lsh as i64),
                sll_b2k: _mm_cvtsi64_si128((64 - base2k) as i64),
                sra_b2k: 64 - base2k,
                srl_b2k: _mm_cvtsi64_si128(base2k as i64),
                b2k: base2k,
                msb: _mm256_set1_epi64x(i64::MIN),
                zero: _mm256_setzero_si256(),
            }
        }
    }
}

/// Shared inner loop body for `nfc_middle_step_avx2` and `nfc_middle_step_assign_avx2`.
///
/// Given deinterleaved split-i128 input `(lo_a, hi_a)` and carry `(lo_c, hi_c)`,
/// returns `(lo_out, new_lo_c, new_hi_c)`.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn nfc_middle_chunk(
    s: &NfcShifts,
    lo_a: __m256i,
    hi_a: __m256i,
    lo_c: __m256i,
    hi_c: __m256i,
) -> (__m256i, __m256i, __m256i) {
    unsafe {
        // digit = get_digit_i128(base2k_lsh, a)
        let lo_dig = sra_epi64(_mm256_sll_epi64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        let hi_dig = sra_epi64(lo_dig, 63);

        // co = (a - digit) >> base2k_lsh
        let diff_lo = _mm256_sub_epi64(lo_a, lo_dig);
        let borrow = _mm256_sub_epi64(
            s.zero,
            _mm256_cmpgt_epi64(_mm256_xor_si256(lo_dig, s.msb), _mm256_xor_si256(lo_a, s.msb)),
        );
        let diff_hi = _mm256_sub_epi64(_mm256_sub_epi64(hi_a, hi_dig), borrow);
        let co_lo = _mm256_or_si256(
            _mm256_srl_epi64(diff_lo, s.srl_b2klsh),
            _mm256_sll_epi64(diff_hi, s.sll_b2klsh),
        );
        let co_hi = sra_epi64(diff_hi, s.b2klsh);

        // digit_shifted = digit << lsh
        let lo_dig_sh = _mm256_sll_epi64(lo_dig, s.sll_lsh);
        let hi_dig_sh = sra_epi64(lo_dig_sh, 63);

        // d_plus_c = digit_shifted + carry
        let lo_dpc = _mm256_add_epi64(lo_dig_sh, lo_c);
        let carry1 = _mm256_sub_epi64(
            s.zero,
            _mm256_cmpgt_epi64(_mm256_xor_si256(lo_dig_sh, s.msb), _mm256_xor_si256(lo_dpc, s.msb)),
        );
        let hi_dpc = _mm256_add_epi64(_mm256_add_epi64(hi_dig_sh, hi_c), carry1);

        // out = get_digit_i128(base2k, d_plus_c)
        let lo_out = sra_epi64(_mm256_sll_epi64(lo_dpc, s.sll_b2k), s.sra_b2k);
        let hi_out = sra_epi64(lo_out, 63);

        // carry2 = (d_plus_c - out) >> base2k
        let diff2_lo = _mm256_sub_epi64(lo_dpc, lo_out);
        let borrow2 = _mm256_sub_epi64(
            s.zero,
            _mm256_cmpgt_epi64(_mm256_xor_si256(lo_out, s.msb), _mm256_xor_si256(lo_dpc, s.msb)),
        );
        let diff2_hi = _mm256_sub_epi64(_mm256_sub_epi64(hi_dpc, hi_out), borrow2);
        let carry2_lo = _mm256_or_si256(_mm256_srl_epi64(diff2_lo, s.srl_b2k), _mm256_sll_epi64(diff2_hi, s.sll_b2k));
        let carry2_hi = sra_epi64(diff2_hi, s.b2k);

        // new_carry = co + carry2
        let new_lo_c = _mm256_add_epi64(co_lo, carry2_lo);
        let carry2 = _mm256_sub_epi64(
            s.zero,
            _mm256_cmpgt_epi64(_mm256_xor_si256(co_lo, s.msb), _mm256_xor_si256(new_lo_c, s.msb)),
        );
        let new_hi_c = _mm256_add_epi64(_mm256_add_epi64(co_hi, carry2_hi), carry2);

        (lo_out, new_lo_c, new_hi_c)
    }
}

/// Inner loop body for `nfc_final_step_assign_avx2`.
///
/// Given deinterleaved `lo_a` (sign-extended i64) and carry `lo_c` (low half only),
/// returns `lo_out`.
///
/// Note: `get_digit(base2k, d_plus_c)` with `base2k ≤ 64` depends only on the low
/// 64 bits of `d_plus_c`, so `hi_dpc` is never needed.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn nfc_final_chunk(s: &NfcShifts, lo_a: __m256i, lo_c: __m256i) -> __m256i {
    unsafe {
        // digit = get_digit_i128(base2k_lsh, r) — lo only since ri is i64.
        let lo_dig = sra_epi64(_mm256_sll_epi64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        // d_plus_c lo = (digit << lsh) + carry_lo
        let lo_dpc = _mm256_add_epi64(_mm256_sll_epi64(lo_dig, s.sll_lsh), lo_c);
        // out = get_digit_i128(base2k, d_plus_c)  — lo only since base2k <= 64
        sra_epi64(_mm256_sll_epi64(lo_dpc, s.sll_b2k), s.sra_b2k)
    }
}

/// AVX2 kernel for `nfc_middle_step` — `i128` input + carry → `i64` output.
///
/// Processes `n` elements with a scalar tail for `n % 4 != 0`.  Handles both
/// `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.  `base2k` must
/// be ≤ 64 (caller is responsible for this precondition).
///
/// # Safety
/// Requires AVX2.  `res`, `a`, `carry` must each have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn nfc_middle_step_avx2(base2k: u32, lsh: u32, n: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts::new(base2k, lsh);
        let a_ptr = a.as_ptr() as *const __m256i;
        let c_ptr = carry.as_mut_ptr() as *mut __m256i;
        let r_ptr = res.as_mut_ptr();

        let chunks = n / 4;
        for i in 0..chunks {
            let a01 = _mm256_loadu_si256(a_ptr.add(2 * i));
            let a23 = _mm256_loadu_si256(a_ptr.add(2 * i + 1));
            let lo_a = _mm256_unpacklo_epi64(a01, a23);
            let hi_a = _mm256_unpackhi_epi64(a01, a23);

            let c01 = _mm256_loadu_si256(c_ptr.add(2 * i));
            let c23 = _mm256_loadu_si256(c_ptr.add(2 * i + 1));
            let lo_c = _mm256_unpacklo_epi64(c01, c23);
            let hi_c = _mm256_unpackhi_epi64(c01, c23);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk(&s, lo_a, hi_a, lo_c, hi_c);

            _mm256_storeu_si256(r_ptr.add(4 * i) as *mut __m256i, _mm256_permute4x64_epi64(lo_out, 0xD8));
            _mm256_storeu_si256(c_ptr.add(2 * i), _mm256_unpacklo_epi64(new_lo_c, new_hi_c));
            _mm256_storeu_si256(c_ptr.add(2 * i + 1), _mm256_unpackhi_epi64(new_lo_c, new_hi_c));
        }

        let tail = chunks * 4;
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

/// AVX2 kernel for `nfc_middle_step_assign` — in-place update of `i64` `res` with `i128` carry.
///
/// Like `nfc_middle_step_avx2` but the input `ai = *r as i128` is read from `res` itself.
/// Handles both `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn nfc_middle_step_assign_avx2(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts::new(base2k, lsh);
        let c_ptr = carry.as_mut_ptr() as *mut __m256i;
        let r_ptr = res.as_mut_ptr();

        let chunks = n / 4;
        for i in 0..chunks {
            // ai = *r as i128; sign-extend i64 to split i128 form.
            let lo_a = _mm256_permute4x64_epi64(_mm256_loadu_si256(r_ptr.add(4 * i) as *const __m256i), 0xD8);
            let hi_a = sra_epi64(lo_a, 63);

            let c01 = _mm256_loadu_si256(c_ptr.add(2 * i));
            let c23 = _mm256_loadu_si256(c_ptr.add(2 * i + 1));
            let lo_c = _mm256_unpacklo_epi64(c01, c23);
            let hi_c = _mm256_unpackhi_epi64(c01, c23);

            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk(&s, lo_a, hi_a, lo_c, hi_c);

            _mm256_storeu_si256(r_ptr.add(4 * i) as *mut __m256i, _mm256_permute4x64_epi64(lo_out, 0xD8));
            _mm256_storeu_si256(c_ptr.add(2 * i), _mm256_unpacklo_epi64(new_lo_c, new_hi_c));
            _mm256_storeu_si256(c_ptr.add(2 * i + 1), _mm256_unpackhi_epi64(new_lo_c, new_hi_c));
        }

        let tail = chunks * 4;
        if tail < n {
            nfc_middle_step_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

/// AVX2 kernel for `nfc_final_step_assign` — flush `i128` carry into the last `i64` limb.
///
/// Computes `*r = get_digit(base2k, (get_digit(base2k_lsh, ri) << lsh) + carry)`.
/// Handles both `lsh == 0` and `lsh != 0` via `base2k_lsh = base2k - lsh`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn nfc_final_step_assign_avx2(base2k: u32, lsh: u32, n: usize, res: &mut [i64], carry: &mut [i128]) {
    unsafe {
        let s = NfcShifts::new(base2k, lsh);
        let c_ptr = carry.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr();

        let chunks = n / 4;
        for i in 0..chunks {
            let lo_a = _mm256_permute4x64_epi64(_mm256_loadu_si256(r_ptr.add(4 * i) as *const __m256i), 0xD8);
            let lo_c = _mm256_unpacklo_epi64(_mm256_loadu_si256(c_ptr.add(2 * i)), _mm256_loadu_si256(c_ptr.add(2 * i + 1)));
            let lo_out = nfc_final_chunk(&s, lo_a, lo_c);
            _mm256_storeu_si256(r_ptr.add(4 * i) as *mut __m256i, _mm256_permute4x64_epi64(lo_out, 0xD8));
        }

        let tail = chunks * 4;
        if tail < n {
            nfc_final_step_assign_scalar(base2k as usize, lsh as usize, &mut res[tail..], &mut carry[tail..]);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice i128 arithmetic — Phase 2 VecZnxBig AVX2
//
// All kernels process 4 × i128 elements per chunk in deinterleaved split
// (lo, hi) form.  A scalar tail handles n % 4 != 0.
//
// Layout: each i128 is stored as [lo: u64, hi: i64] (little-endian x86-64).
// Loading two consecutive __m256i gives [lo0,hi0,lo1,hi1] and [lo2,hi2,lo3,hi3].
// Deinterleave with unpacklo/unpackhi: lo=[lo0,lo2,lo1,lo3], hi=[hi0,hi2,hi1,hi3].
// After computation, reinterleave with unpacklo/unpackhi before storing.
// ──────────────────────────────────────────────────────────────────────────────

/// Load 4 i128s from `a_ptr[2*i .. 2*i+2]` into deinterleaved split form `(lo, hi)`.
///
/// # Safety
/// Requires AVX2.  `a_ptr` must be valid for at least `2*(i+1)` × `__m256i` loads.
#[inline(always)]
unsafe fn load4_i128(a_ptr: *const __m256i, i: usize) -> (__m256i, __m256i) {
    unsafe {
        let a01 = _mm256_loadu_si256(a_ptr.add(2 * i));
        let a23 = _mm256_loadu_si256(a_ptr.add(2 * i + 1));
        let lo = _mm256_unpacklo_epi64(a01, a23); // [lo0, lo2, lo1, lo3]
        let hi = _mm256_unpackhi_epi64(a01, a23); // [hi0, hi2, hi1, hi3]
        (lo, hi)
    }
}

/// Load 4 i64s from `a_ptr[i]` (one `__m256i`) and sign-extend to split i128 form `(lo, hi)`.
///
/// # Safety
/// Requires AVX2.  `a_ptr` must be valid for at least `i+1` × `__m256i` loads.
#[inline(always)]
unsafe fn load4_i64_as_i128(a_ptr: *const __m256i, i: usize) -> (__m256i, __m256i) {
    unsafe {
        let a_vec = _mm256_loadu_si256(a_ptr.add(i));
        let lo = _mm256_permute4x64_epi64(a_vec, 0xD8); // [a0, a2, a1, a3]
        let hi = sra_epi64(lo, 63); // sign broadcast
        (lo, hi)
    }
}

/// Store 4 i128s in deinterleaved split form `(lo_r, hi_r)` to `r_ptr[2*i .. 2*i+2]`.
///
/// # Safety
/// Requires AVX2.  `r_ptr` must be valid for at least `2*(i+1)` × `__m256i` stores.
#[inline(always)]
unsafe fn store4_i128(r_ptr: *mut __m256i, i: usize, lo_r: __m256i, hi_r: __m256i) {
    unsafe {
        _mm256_storeu_si256(r_ptr.add(2 * i), _mm256_unpacklo_epi64(lo_r, hi_r));
        _mm256_storeu_si256(r_ptr.add(2 * i + 1), _mm256_unpackhi_epi64(lo_r, hi_r));
    }
}

/// 128-bit addition in split form: `(lo_r, hi_r) = (lo_a + lo_b, hi_a + hi_b + carry)`.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn add4_i128(lo_a: __m256i, hi_a: __m256i, lo_b: __m256i, hi_b: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let msb = _mm256_set1_epi64x(i64::MIN);
        let zero = _mm256_setzero_si256();
        let lo_r = _mm256_add_epi64(lo_a, lo_b);
        // carry: lo_a >_u lo_r  ⟺  unsigned addition overflowed
        let carry_mask = _mm256_cmpgt_epi64(_mm256_xor_si256(lo_a, msb), _mm256_xor_si256(lo_r, msb));
        let carry_one = _mm256_sub_epi64(zero, carry_mask); // 0 or 1
        let hi_r = _mm256_add_epi64(_mm256_add_epi64(hi_a, hi_b), carry_one);
        (lo_r, hi_r)
    }
}

/// 128-bit subtraction in split form: `(lo_r, hi_r) = (lo_a - lo_b, hi_a - hi_b - borrow)`.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn sub4_i128(lo_a: __m256i, hi_a: __m256i, lo_b: __m256i, hi_b: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let msb = _mm256_set1_epi64x(i64::MIN);
        let zero = _mm256_setzero_si256();
        let lo_r = _mm256_sub_epi64(lo_a, lo_b);
        // borrow: lo_b >_u lo_a  ⟺  unsigned subtraction underflowed
        let borrow_mask = _mm256_cmpgt_epi64(_mm256_xor_si256(lo_b, msb), _mm256_xor_si256(lo_a, msb));
        let borrow_one = _mm256_sub_epi64(zero, borrow_mask); // 0 or 1
        let hi_r = _mm256_sub_epi64(_mm256_sub_epi64(hi_a, hi_b), borrow_one);
        (lo_r, hi_r)
    }
}

/// 128-bit negation in split form: `(lo_r, hi_r) = -(lo_a, hi_a)`.
///
/// Uses the identity `-(lo, hi) = (~hi + carry, 0 - lo)` where `carry = (lo == 0) ? 1 : 0`.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn neg4_i128(lo_a: __m256i, hi_a: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let zero = _mm256_setzero_si256();
        let all_ones = _mm256_cmpeq_epi64(zero, zero); // every bit set
        let lo_r = _mm256_sub_epi64(zero, lo_a);
        // carry into hi = 1 iff lo_a == 0  (two's-complement negation carry)
        let carry_mask = _mm256_cmpeq_epi64(lo_a, zero); // all-ones when lo_a == 0
        let carry_one = _mm256_sub_epi64(zero, carry_mask); // 0 - (-1) = 1  or  0 - 0 = 0
        // hi_r = ~hi_a + carry  =  xor(hi_a, all_ones) + carry_one
        let hi_r = _mm256_add_epi64(_mm256_xor_si256(hi_a, all_ones), carry_one);
        (lo_r, hi_r)
    }
}

/// `res[i] = a[i].wrapping_add(b[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_add_avx2(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_b, hi_b) = load4_i128(b_ptr, i);
            let (lo_r, hi_r) = add4_i128(lo_a, hi_a, lo_b, hi_b);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
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
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_add_assign_avx2(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_r, hi_r) = add4_i128(lo_r, hi_r, lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai));
    }
}

/// `res[i] = a[i].wrapping_add(b[i] as i128)` for `n` elements (`b` is `i64`).
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_add_small_avx2(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_b, hi_b) = load4_i64_as_i128(b_ptr, i);
            let (lo_r, hi_r) = add4_i128(lo_a, hi_a, lo_b, hi_b);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
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
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_add_small_assign_avx2(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = add4_i128(lo_r, hi_r, lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai as i128));
    }
}

/// `res[i] = a[i].wrapping_sub(b[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_avx2(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_b, hi_b) = load4_i128(b_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_a, hi_a, lo_b, hi_b);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
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
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_assign_avx2(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_r, hi_r, lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai));
    }
}

/// `res[i] = a[i].wrapping_sub(res[i])` for `n` i128 elements.
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_negate_assign_avx2(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_a, hi_a, lo_r, hi_r); // a − res
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = ai.wrapping_sub(*r));
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(b[i])` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_small_a_avx2(n: usize, res: &mut [i128], a: &[i64], b: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            let (lo_b, hi_b) = load4_i128(b_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_a, hi_a, lo_b, hi_b);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
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
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_small_b_avx2(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let b_ptr = b.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_b, hi_b) = load4_i64_as_i128(b_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_a, hi_a, lo_b, hi_b);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
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
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_small_assign_avx2(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_r, hi_r, lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai as i128));
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(res[i])` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_sub_small_negate_assign_avx2(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = sub4_i128(lo_a, hi_a, lo_r, hi_r); // a − res
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = (ai as i128).wrapping_sub(*r));
    }
}

/// `res[i] = a[i].wrapping_neg()` for `n` i128 elements.
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_negate_avx2(n: usize, res: &mut [i128], a: &[i128]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i128(a_ptr, i);
            let (lo_r, hi_r) = neg4_i128(lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = ai.wrapping_neg());
    }
}

/// `res[i] = res[i].wrapping_neg()` for `n` i128 elements.
///
/// # Safety
/// Requires AVX2.  Slice must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_negate_assign_avx2(n: usize, res: &mut [i128]) {
    unsafe {
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_r, hi_r) = load4_i128(r_ptr as *const __m256i, i);
            let (lo_r, hi_r) = neg4_i128(lo_r, hi_r);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n].iter_mut().for_each(|r| *r = r.wrapping_neg());
    }
}

/// `res[i] = a[i] as i128` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_from_small_avx2(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            store4_i128(r_ptr, i, lo_a, hi_a);
        }
        let tail = chunks * 4;
        for j in tail..n {
            res[j] = a[j] as i128;
        }
    }
}

/// `res[i] = -(a[i] as i128)` for `n` elements (`a` is `i64`).
///
/// # Safety
/// Requires AVX2.  All slices must have at least `n` elements.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vi128_neg_from_small_avx2(n: usize, res: &mut [i128], a: &[i64]) {
    unsafe {
        let a_ptr = a.as_ptr() as *const __m256i;
        let r_ptr = res.as_mut_ptr() as *mut __m256i;
        let chunks = n / 4;
        for i in 0..chunks {
            let (lo_a, hi_a) = load4_i64_as_i128(a_ptr, i);
            let (lo_r, hi_r) = neg4_i128(lo_a, hi_a);
            store4_i128(r_ptr, i, lo_r, hi_r);
        }
        let tail = chunks * 4;
        res[tail..n]
            .iter_mut()
            .zip(a[tail..n].iter())
            .for_each(|(r, &ai)| *r = -(ai as i128));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use super::{
        nfc_final_step_assign_avx2, nfc_final_step_assign_scalar, nfc_middle_step_assign_avx2, nfc_middle_step_assign_scalar,
        nfc_middle_step_avx2, nfc_middle_step_scalar, vi128_add_avx2, vi128_from_small_avx2, vi128_neg_from_small_avx2,
        vi128_negate_avx2, vi128_sub_avx2,
    };

    fn i128_data(n: usize, seed: i128) -> Vec<i128> {
        (0..n).map(|i| (i as i128 * seed + seed / 3) % (1i128 << 80)).collect()
    }

    fn i64_data(n: usize, seed: i64) -> Vec<i64> {
        (0..n).map(|i| i as i64 * seed - seed / 2).collect()
    }

    #[test]
    fn vi128_add_avx2_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x1_0000_0001i128);
        let b = i128_data(n, 0x0_FFFF_FFFFi128);
        let expected: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_add_avx2(n, &mut res, &a, &b) };
        assert_eq!(res, expected, "vi128_add_avx2 mismatch");
    }

    #[test]
    fn vi128_sub_avx2_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x2_0000_0003i128);
        let b = i128_data(n, 0x1_0000_0001i128);
        let expected: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_sub_avx2(n, &mut res, &a, &b) };
        assert_eq!(res, expected, "vi128_sub_avx2 mismatch");
    }

    #[test]
    fn vi128_negate_avx2_vs_scalar() {
        let n = 64usize;
        let a = i128_data(n, 0x1_2345_6789i128);
        let expected: Vec<i128> = a.iter().map(|x| -x).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_negate_avx2(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_negate_avx2 mismatch");
    }

    #[test]
    fn vi128_from_small_avx2_vs_scalar() {
        let n = 64usize;
        let a = i64_data(n, 12345);
        let expected: Vec<i128> = a.iter().map(|&x| x as i128).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_from_small_avx2(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_from_small_avx2 mismatch");
    }

    #[test]
    fn vi128_neg_from_small_avx2_vs_scalar() {
        let n = 64usize;
        let a = i64_data(n, 99);
        let expected: Vec<i128> = a.iter().map(|&x| -(x as i128)).collect();

        let mut res = vec![0i128; n];
        unsafe { vi128_neg_from_small_avx2(n, &mut res, &a) };
        assert_eq!(res, expected, "vi128_neg_from_small_avx2 mismatch");
    }

    #[test]
    fn nfc_middle_step_avx2_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 0usize;
        let a = i128_data(n, 37i128);
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 3) % (1i128 << 20)).collect();

        let mut res_avx = vec![0i64; n];
        let mut carry_avx = carry_init.clone();
        let mut res_ref = vec![0i64; n];
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_middle_step_avx2(base2k as u32, lsh as u32, n, &mut res_avx, &a, &mut carry_avx) };
        nfc_middle_step_scalar(base2k, lsh, &mut res_ref, &a, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_middle_step res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_middle_step carry mismatch");
    }

    #[test]
    fn nfc_middle_step_assign_avx2_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 8usize;
        let init: Vec<i64> = (0..n).map(|i| (i as i64 * 5) % (1i64 << 20)).collect();
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 7) % (1i128 << 20)).collect();

        let mut res_avx = init.clone();
        let mut carry_avx = carry_init.clone();
        let mut res_ref = init.clone();
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_middle_step_assign_avx2(base2k as u32, lsh as u32, n, &mut res_avx, &mut carry_avx) };
        nfc_middle_step_assign_scalar(base2k, lsh, &mut res_ref, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_middle_step_assign res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_middle_step_assign carry mismatch");
    }

    #[test]
    fn nfc_final_step_assign_avx2_vs_scalar() {
        let n = 64usize;
        let base2k = 16usize;
        let lsh = 0usize;
        let init: Vec<i64> = (0..n).map(|i| (i as i64 * 3) % (1i64 << 20)).collect();
        let carry_init: Vec<i128> = (0..n).map(|i| (i as i128 * 11) % (1i128 << 20)).collect();

        let mut res_avx = init.clone();
        let mut carry_avx = carry_init.clone();
        let mut res_ref = init.clone();
        let mut carry_ref = carry_init.clone();

        unsafe { nfc_final_step_assign_avx2(base2k as u32, lsh as u32, n, &mut res_avx, &mut carry_avx) };
        nfc_final_step_assign_scalar(base2k, lsh, &mut res_ref, &mut carry_ref);

        assert_eq!(res_avx, res_ref, "nfc_final_step_assign res mismatch");
        assert_eq!(carry_avx, carry_ref, "nfc_final_step_assign carry mismatch");
    }
}
