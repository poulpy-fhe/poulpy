//! Raw AVX512-IFMA forward and inverse NTT kernels.
//!
//! These kernels are the core arithmetic engine of the IFMA backend.
//!
//! - Butterfly values live in `[0, 4q)`; a single final pass renormalises to `[0, 2q)`.
//! - Diff path feeds directly into Harvey without a pre-reduction (IFMA's 52-bit
//!   product absorbs inputs up to `2^52`).
//! - Harvey multiplication replaces the AVX2 split-precomputed multiply path.
//! - Two coefficients are processed at a time through 512-bit loads where profitable.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64,
    _mm256_min_epu64, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi64, _mm512_add_epi64,
    _mm512_and_si512, _mm512_castsi256_si512, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_min_epu64, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi64,
};

use std::mem::size_of;

use crate::ntt126_ifma::{
    primes::PrimeSetNtt126Ifma,
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD arithmetic primitives
// ──────────────────────────────────────────────────────────────────────────────

/// Conditional subtract of `q2`: if x >= q2 (unsigned), return x - q2, else x.
///
/// Uses the `min_epu64` identity: `min(x, x − q2 mod 2^64) == x − q2` when
/// `x ≥ q2` (no underflow so the wrapped difference is smaller than x) and
/// `== x` when `x < q2` (the wrapped difference is huge and `x` wins).
/// This is 2 µops vs 4 for the MSB-flip / cmpgt idiom.
#[inline]
#[target_feature(enable = "avx512vl")]
pub(crate) unsafe fn cond_sub_2q_si256(x: __m256i, q2: __m256i) -> __m256i {
    let diff = _mm256_sub_epi64(x, q2);
    _mm256_min_epu64(x, diff)
}

/// Harvey modular multiply — 4 lanes.
///
/// Input: `a ∈ [0, 2^52)` (in practice up to `8q` under lazy reduction),
/// `omega ∈ [0, q)`.  Output: `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`.
///
/// Since `r = a·ω − qhat·q ∈ [0, 2q) ⊂ [0, 2^52)`, we only need the low-52
/// bits of `a·ω` and `qhat·q`. Reconstructing full 64-bit products (mask +
/// shift + add) is wasted work; `madd52lo` alone suffices, and a final mask
/// to 52 bits handles the borrow case (when `lo52(a·ω) < lo52(qhat·q)` even
/// though the mathematical difference is non-negative).
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn harvey_modmul_si256(a: __m256i, omega: __m256i, omega_quot: __m256i, q: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let mask52 = _mm256_set1_epi64x((1i64 << 52) - 1);
    let qhat = _mm256_madd52hi_epu64(zero, a, omega_quot);
    let prod_lo52 = _mm256_madd52lo_epu64(zero, a, omega);
    let qq_lo52 = _mm256_madd52lo_epu64(zero, qhat, q);
    _mm256_and_si256(_mm256_sub_epi64(prod_lo52, qq_lo52), mask52)
}

// ──────────────────────────────────────────────────────────────────────────────
// 512-bit wide primitives (2 CRT coefficients per __m512i)
// ──────────────────────────────────────────────────────────────────────────────

/// Conditional subtract of `q2` on 8 lanes (2 coefficients).
///
/// See [`cond_sub_2q_si256`] for the `min_epu64` trick rationale.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cond_sub_2q_si512(x: __m512i, q2: __m512i) -> __m512i {
    let diff = _mm512_sub_epi64(x, q2);
    _mm512_min_epu64(x, diff)
}

/// Harvey modular multiply — 8 lanes (2 coefficients).
///
/// Identical low-52 trick as the 256-bit variant: 3 IFMA + sub + mask.
#[inline]
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn harvey_modmul_si512(a: __m512i, omega: __m512i, omega_quot: __m512i, q: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let mask52 = _mm512_set1_epi64((1i64 << 52) - 1);
    let qhat = _mm512_madd52hi_epu64(zero, a, omega_quot);
    let prod_lo52 = _mm512_madd52lo_epu64(zero, a, omega);
    let qq_lo52 = _mm512_madd52lo_epu64(zero, qhat, q);
    _mm512_and_si512(_mm512_sub_epi64(prod_lo52, qq_lo52), mask52)
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT butterfly kernels
// ──────────────────────────────────────────────────────────────────────────────

/// Pack two consecutive `__m256i` values into one `__m512i`.
#[inline(always)]
unsafe fn pack_512(lo: __m256i, hi: __m256i) -> __m512i {
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi) }
}

/// Level-0: `a[i] *= ω^i` using Harvey multiply.
/// Uses 512-bit main loop with split twiddle layout.
///
/// `po_omega`: pointer to contiguous ω values for this segment.
/// `po_quot`: pointer to contiguous ωq values for this segment.
#[target_feature(enable = "avx512ifma")]
unsafe fn ntt_iter_first_ifma(
    begin: *mut __m256i,
    end: *const __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
    q: __m256i,
) {
    unsafe {
        let q_512 = pack_512(q, q);
        let n_coeffs = (end as usize - begin as usize) / size_of::<__m256i>();

        // 512-bit main loop: 2 coefficients at a time — single 512-bit load per twiddle
        let pairs = n_coeffs / 2;
        let data_512 = begin as *mut __m512i;
        let omega_512 = po_omega as *const __m512i;
        let quot_512 = po_quot as *const __m512i;
        let unrolled_pairs = pairs / 2;
        for i in 0..unrolled_pairs {
            let base = i * 2;
            let x0 = _mm512_loadu_si512(data_512.add(base));
            let omega0 = _mm512_loadu_si512(omega_512.add(base));
            let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
            let x1 = _mm512_loadu_si512(data_512.add(base + 1));
            let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
            let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
            _mm512_storeu_si512(data_512.add(base), harvey_modmul_si512(x0, omega0, omega_quot0, q_512));
            _mm512_storeu_si512(data_512.add(base + 1), harvey_modmul_si512(x1, omega1, omega_quot1, q_512));
        }

        if !pairs.is_multiple_of(2) {
            let i = pairs - 1;
            let x = _mm512_loadu_si512(data_512.add(i));
            let omega = _mm512_loadu_si512(omega_512.add(i));
            let omega_quot = _mm512_loadu_si512(quot_512.add(i));
            _mm512_storeu_si512(data_512.add(i), harvey_modmul_si512(x, omega, omega_quot, q_512));
        }

        // 256-bit tail
        if !n_coeffs.is_multiple_of(2) {
            let idx = n_coeffs - 1;
            let x = _mm256_loadu_si256(begin.add(idx));
            let omega = _mm256_loadu_si256(po_omega.add(idx));
            let omega_quot = _mm256_loadu_si256(po_quot.add(idx));
            _mm256_storeu_si256(begin.add(idx), harvey_modmul_si256(x, omega, omega_quot, q));
        }
    }
}

/// Forward Cooley-Tukey butterfly with IFMA-native lazy arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
///
/// All inputs and outputs in `[0, 4q)`.  Sum path subtracts `4q`; diff path
/// is fed directly into the Harvey multiply (which absorbs the reduction).
#[target_feature(enable = "avx512ifma")]
#[inline]
unsafe fn ntt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q4: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q4_512 = pack_512(q4, q4);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle (both sides use cond_sub_4q)
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
            }

            // i = 1..halfnn-1: diff fed directly into Harvey (split layout).
            // Peel i=1 so the wide data loop starts on a 64-byte boundary.
            let remaining = halfnn - 1;
            let twiddle_shift = if remaining > 0 {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = _mm256_sub_epi64(_mm256_add_epi64(a, q4), b);
                let omega = _mm256_loadu_si256(po_omega);
                let omega_quot = _mm256_loadu_si256(po_quot);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, harvey_modmul_si256(diff, omega, omega_quot, q));
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                1
            } else {
                0
            };
            let remaining = remaining - twiddle_shift;

            // 512-bit pairs, unrolled to expose independent Harvey chains.
            let pairs = remaining / 2;
            let omega_512 = po_omega.add(twiddle_shift) as *const __m512i;
            let quot_512 = po_quot.add(twiddle_shift) as *const __m512i;
            let quads = pairs / 4;
            for p in 0..quads {
                let base = p * 4;

                let av0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let av2 = _mm512_loadu_si512(ptr1.add(4) as *const __m512i);
                let bv2 = _mm512_loadu_si512(ptr2.add(4) as *const __m512i);
                let av3 = _mm512_loadu_si512(ptr1.add(6) as *const __m512i);
                let bv3 = _mm512_loadu_si512(ptr2.add(6) as *const __m512i);

                let omega0 = _mm512_loadu_si512(omega_512.add(base));
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega2 = _mm512_loadu_si512(omega_512.add(base + 2));
                let omega3 = _mm512_loadu_si512(omega_512.add(base + 3));
                let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let omega_quot2 = _mm512_loadu_si512(quot_512.add(base + 2));
                let omega_quot3 = _mm512_loadu_si512(quot_512.add(base + 3));

                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bv0), q4_512);
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bv1), q4_512);
                let sum2 = cond_sub_2q_si512(_mm512_add_epi64(av2, bv2), q4_512);
                let sum3 = cond_sub_2q_si512(_mm512_add_epi64(av3, bv3), q4_512);

                let diff0 = _mm512_sub_epi64(_mm512_add_epi64(av0, q4_512), bv0);
                let diff1 = _mm512_sub_epi64(_mm512_add_epi64(av1, q4_512), bv1);
                let diff2 = _mm512_sub_epi64(_mm512_add_epi64(av2, q4_512), bv2);
                let diff3 = _mm512_sub_epi64(_mm512_add_epi64(av3, q4_512), bv3);

                let out0 = harvey_modmul_si512(diff0, omega0, omega_quot0, q_512);
                let out1 = harvey_modmul_si512(diff1, omega1, omega_quot1, q_512);
                let out2 = harvey_modmul_si512(diff2, omega2, omega_quot2, q_512);
                let out3 = harvey_modmul_si512(diff3, omega3, omega_quot3, q_512);

                _mm512_storeu_si512(ptr1 as *mut __m512i, sum0);
                _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, sum1);
                _mm512_storeu_si512(ptr1.add(4) as *mut __m512i, sum2);
                _mm512_storeu_si512(ptr1.add(6) as *mut __m512i, sum3);
                _mm512_storeu_si512(ptr2 as *mut __m512i, out0);
                _mm512_storeu_si512(ptr2.add(2) as *mut __m512i, out1);
                _mm512_storeu_si512(ptr2.add(4) as *mut __m512i, out2);
                _mm512_storeu_si512(ptr2.add(6) as *mut __m512i, out3);

                ptr1 = ptr1.add(8);
                ptr2 = ptr2.add(8);
            }

            // Pair-at-a-time tail (0..3 remaining pairs).
            let mut p_idx = quads * 4;
            while p_idx < pairs {
                let av = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv = _mm512_loadu_si512(ptr2 as *const __m512i);
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bv), q4_512);
                let diff = _mm512_sub_epi64(_mm512_add_epi64(av, q4_512), bv);
                let omega = _mm512_loadu_si512(omega_512.add(p_idx));
                let omega_quot = _mm512_loadu_si512(quot_512.add(p_idx));
                _mm512_storeu_si512(ptr1 as *mut __m512i, sum);
                _mm512_storeu_si512(ptr2 as *mut __m512i, harvey_modmul_si512(diff, omega, omega_quot, q_512));
                ptr1 = ptr1.add(2);
                ptr2 = ptr2.add(2);
                p_idx += 1;
            }

            // 256-bit tail
            if !remaining.is_multiple_of(2) {
                let tail_idx = twiddle_shift + pairs * 2;
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = _mm256_sub_epi64(_mm256_add_epi64(a, q4), b);
                let omega = _mm256_loadu_si256(po_omega.add(tail_idx));
                let omega_quot = _mm256_loadu_si256(po_quot.add(tail_idx));
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, harvey_modmul_si256(diff, omega, omega_quot, q));
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman-Sande butterfly with IFMA-native lazy arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
///
/// All inputs and outputs in `[0, 4q)`.  `b_raw ∈ [0, 4q)` is fed directly into
/// Harvey (output `∈ [0, 2q)`); sum/diff use `cond_sub_4q`.
#[target_feature(enable = "avx512ifma")]
#[inline]
unsafe fn intt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q4: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q4_512 = pack_512(q4, q4);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
            }

            // Peel i=1 so the wide data loop starts on a 64-byte boundary.
            let remaining = halfnn - 1;
            let twiddle_shift = if remaining > 0 {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let omega = _mm256_loadu_si256(po_omega);
                let omega_quot = _mm256_loadu_si256(po_quot);
                let bo = harvey_modmul_si256(b, omega, omega_quot, q);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, bo), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), bo), q4);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                1
            } else {
                0
            };
            let remaining = remaining - twiddle_shift;

            // 512-bit pairs, unrolled to expose independent Harvey chains.
            let pairs = remaining / 2;
            let omega_512 = po_omega.add(twiddle_shift) as *const __m512i;
            let quot_512 = po_quot.add(twiddle_shift) as *const __m512i;
            let quads = pairs / 4;
            for p in 0..quads {
                let base = p * 4;

                let bv0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let bv2 = _mm512_loadu_si512(ptr2.add(4) as *const __m512i);
                let bv3 = _mm512_loadu_si512(ptr2.add(6) as *const __m512i);

                let omega0 = _mm512_loadu_si512(omega_512.add(base));
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega2 = _mm512_loadu_si512(omega_512.add(base + 2));
                let omega3 = _mm512_loadu_si512(omega_512.add(base + 3));
                let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let omega_quot2 = _mm512_loadu_si512(quot_512.add(base + 2));
                let omega_quot3 = _mm512_loadu_si512(quot_512.add(base + 3));

                let bo0 = harvey_modmul_si512(bv0, omega0, omega_quot0, q_512);
                let bo1 = harvey_modmul_si512(bv1, omega1, omega_quot1, q_512);
                let bo2 = harvey_modmul_si512(bv2, omega2, omega_quot2, q_512);
                let bo3 = harvey_modmul_si512(bv3, omega3, omega_quot3, q_512);

                let av0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let av2 = _mm512_loadu_si512(ptr1.add(4) as *const __m512i);
                let av3 = _mm512_loadu_si512(ptr1.add(6) as *const __m512i);

                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bo0), q4_512);
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bo1), q4_512);
                let sum2 = cond_sub_2q_si512(_mm512_add_epi64(av2, bo2), q4_512);
                let sum3 = cond_sub_2q_si512(_mm512_add_epi64(av3, bo3), q4_512);

                let diff0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av0, q4_512), bo0), q4_512);
                let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av1, q4_512), bo1), q4_512);
                let diff2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av2, q4_512), bo2), q4_512);
                let diff3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av3, q4_512), bo3), q4_512);

                _mm512_storeu_si512(ptr1 as *mut __m512i, sum0);
                _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, sum1);
                _mm512_storeu_si512(ptr1.add(4) as *mut __m512i, sum2);
                _mm512_storeu_si512(ptr1.add(6) as *mut __m512i, sum3);
                _mm512_storeu_si512(ptr2 as *mut __m512i, diff0);
                _mm512_storeu_si512(ptr2.add(2) as *mut __m512i, diff1);
                _mm512_storeu_si512(ptr2.add(4) as *mut __m512i, diff2);
                _mm512_storeu_si512(ptr2.add(6) as *mut __m512i, diff3);

                ptr1 = ptr1.add(8);
                ptr2 = ptr2.add(8);
            }

            // Pair-at-a-time tail (0..3 remaining pairs).
            let mut p_idx = quads * 4;
            while p_idx < pairs {
                let av = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv = _mm512_loadu_si512(ptr2 as *const __m512i);
                let omega = _mm512_loadu_si512(omega_512.add(p_idx));
                let omega_quot = _mm512_loadu_si512(quot_512.add(p_idx));
                let bo = harvey_modmul_si512(bv, omega, omega_quot, q_512);
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4_512);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4_512), bo), q4_512);
                _mm512_storeu_si512(ptr1 as *mut __m512i, sum);
                _mm512_storeu_si512(ptr2 as *mut __m512i, diff);
                ptr1 = ptr1.add(2);
                ptr2 = ptr2.add(2);
                p_idx += 1;
            }

            // 256-bit tail
            if !remaining.is_multiple_of(2) {
                let tail_idx = twiddle_shift + pairs * 2;
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let omega = _mm256_loadu_si256(po_omega.add(tail_idx));
                let omega_quot = _mm256_loadu_si256(po_quot.add(tail_idx));
                let bo = harvey_modmul_si256(b, omega, omega_quot, q);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, bo), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), bo), q4);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
            }
            data = data.add(nn);
        }
    }
}

/// Fused iNTT pass covering `nn = 2, 4, 8` in registers.
///
/// Twiddle layout in `po_base` (level-2 has no twiddles):
///   [0]=ω₄ [1]=ω₄ quot | [2..5)=ω₈^{1,2,3} [5..8)=ω₈ quots.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_radix8_first3_ifma(begin: *mut __m256i, end: *const __m256i, q: __m256i, q4: __m256i, po_base: *const __m256i) {
    unsafe {
        let w4 = _mm256_loadu_si256(po_base);
        let w4q = _mm256_loadu_si256(po_base.add(1));
        let w8_1 = _mm256_loadu_si256(po_base.add(2));
        let w8_2 = _mm256_loadu_si256(po_base.add(3));
        let w8_3 = _mm256_loadu_si256(po_base.add(4));
        let w8_1q = _mm256_loadu_si256(po_base.add(5));
        let w8_2q = _mm256_loadu_si256(po_base.add(6));
        let w8_3q = _mm256_loadu_si256(po_base.add(7));

        let mut ptr = begin;
        while (ptr as usize) < (end as usize) {
            let a0 = _mm256_loadu_si256(ptr);
            let a1 = _mm256_loadu_si256(ptr.add(1));
            let a2 = _mm256_loadu_si256(ptr.add(2));
            let a3 = _mm256_loadu_si256(ptr.add(3));
            let a4 = _mm256_loadu_si256(ptr.add(4));
            let a5 = _mm256_loadu_si256(ptr.add(5));
            let a6 = _mm256_loadu_si256(ptr.add(6));
            let a7 = _mm256_loadu_si256(ptr.add(7));

            // nn=2: 4 identity butterflies.
            let t0 = cond_sub_2q_si256(_mm256_add_epi64(a0, a1), q4);
            let t1 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a0, q4), a1), q4);
            let t2 = cond_sub_2q_si256(_mm256_add_epi64(a2, a3), q4);
            let t3 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a2, q4), a3), q4);
            let t4 = cond_sub_2q_si256(_mm256_add_epi64(a4, a5), q4);
            let t5 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a4, q4), a5), q4);
            let t6 = cond_sub_2q_si256(_mm256_add_epi64(a6, a7), q4);
            let t7 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a6, q4), a7), q4);

            // nn=4: identity at i=0, twiddle w4 at i=1, for each half-block.
            let u0 = cond_sub_2q_si256(_mm256_add_epi64(t0, t2), q4);
            let u2 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(t0, q4), t2), q4);
            let bo13 = harvey_modmul_si256(t3, w4, w4q, q);
            let u1 = cond_sub_2q_si256(_mm256_add_epi64(t1, bo13), q4);
            let u3 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(t1, q4), bo13), q4);

            let u4 = cond_sub_2q_si256(_mm256_add_epi64(t4, t6), q4);
            let u6 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(t4, q4), t6), q4);
            let bo57 = harvey_modmul_si256(t7, w4, w4q, q);
            let u5 = cond_sub_2q_si256(_mm256_add_epi64(t5, bo57), q4);
            let u7 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(t5, q4), bo57), q4);

            // nn=8: identity at i=0, twiddles w8_{1,2,3} at i=1..3.
            let v0 = cond_sub_2q_si256(_mm256_add_epi64(u0, u4), q4);
            let v4 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u0, q4), u4), q4);
            let bo15 = harvey_modmul_si256(u5, w8_1, w8_1q, q);
            let v1 = cond_sub_2q_si256(_mm256_add_epi64(u1, bo15), q4);
            let v5 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u1, q4), bo15), q4);
            let bo26 = harvey_modmul_si256(u6, w8_2, w8_2q, q);
            let v2 = cond_sub_2q_si256(_mm256_add_epi64(u2, bo26), q4);
            let v6 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u2, q4), bo26), q4);
            let bo37 = harvey_modmul_si256(u7, w8_3, w8_3q, q);
            let v3 = cond_sub_2q_si256(_mm256_add_epi64(u3, bo37), q4);
            let v7 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u3, q4), bo37), q4);

            _mm256_storeu_si256(ptr, v0);
            _mm256_storeu_si256(ptr.add(1), v1);
            _mm256_storeu_si256(ptr.add(2), v2);
            _mm256_storeu_si256(ptr.add(3), v3);
            _mm256_storeu_si256(ptr.add(4), v4);
            _mm256_storeu_si256(ptr.add(5), v5);
            _mm256_storeu_si256(ptr.add(6), v6);
            _mm256_storeu_si256(ptr.add(7), v7);

            ptr = ptr.add(8);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Algorithmic block size for the blocked NTT. Once butterfly size drops to
/// this many coefficients, we stop sweeping all data per level and instead
/// run the remaining levels fully inside each block before moving on.
///
/// Each block holds `NTT_BLOCK * 32` bytes of data (3-prime CRT, 4 x u64 per
/// coefficient). At `256`, the block is 8 KiB, leaving room in L1d for twiddle
/// streams and other live data without assuming a specific higher-level
/// parameter set.
const NTT_BLOCK: usize = 256;

/// Forward NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// Butterfly values live in `[0, 4q)`; a final pass renormalises to `[0, 2q)`.
/// Butterfly levels larger than `NTT_BLOCK` run breadth-first; inner levels
/// (≤ `NTT_BLOCK`) are performed block-by-block, keeping the working set in
/// cache across all remaining levels.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn ntt_avx512<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let q = {
            let a = P::Q[0];
            let b = P::Q[1];
            let c = P::Q[2];
            use core::arch::x86_64::_mm256_set_epi64x;
            _mm256_set_epi64x(0, c as i64, b as i64, a as i64)
        };
        let q2 = _mm256_loadu_si256(table.q2.as_ptr() as *const __m256i);
        let q4 = _mm256_loadu_si256(table.q4.as_ptr() as *const __m256i);

        let mut seg_avx = 0usize;

        // Level 0: a[i] *= ω^i.
        ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
        seg_avx += 2 * n;

        // Upper butterfly levels (breadth-first) while nn > NTT_BLOCK.
        let block = NTT_BLOCK.min(n);
        let mut nn = n;
        while nn > block {
            let halfnn = nn / 2;
            let count = halfnn - 1;
            ntt_iter_ifma(nn, begin, end, q, q4, po_base.add(seg_avx), po_base.add(seg_avx + count));
            seg_avx += 2 * count;
            nn /= 2;
        }

        // Precompute segment offsets for each remaining level (nn, nn/2, …, 2).
        let mut inner_segs = [0usize; 17];
        let mut inner_nn = [0usize; 17];
        let mut num_inner = 0usize;
        {
            let mut m = nn;
            let mut s = seg_avx;
            while m >= 2 {
                inner_nn[num_inner] = m;
                inner_segs[num_inner] = s;
                let halfm = m / 2;
                if halfm > 1 {
                    s += 2 * (halfm - 1);
                }
                m /= 2;
                num_inner += 1;
            }
        }

        // Inner levels (depth-first by block): run the whole remaining level
        // sequence inside each block before moving to the next.  Each block
        // is `nn = NTT_BLOCK` coefficients; subsequent levels subdivide it.
        let mut blk_start = 0usize;
        while blk_start < n {
            let blk_begin = begin.add(blk_start);
            let blk_end = begin.add(blk_start + nn) as *const __m256i;
            for i in 0..num_inner {
                let m = inner_nn[i];
                let halfm = m / 2;
                let seg = inner_segs[i];
                if halfm > 1 {
                    let count = halfm - 1;
                    ntt_iter_ifma(m, blk_begin, blk_end, q, q4, po_base.add(seg), po_base.add(seg + count));
                } else {
                    // m == 2: add/sub only, no twiddle.
                    let mut p1 = blk_begin;
                    let mut p2 = blk_begin.add(1);
                    while (p1 as usize) < (blk_end as usize) {
                        let a = _mm256_loadu_si256(p1);
                        let b = _mm256_loadu_si256(p2);
                        let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                        let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                        _mm256_storeu_si256(p1, sum);
                        _mm256_storeu_si256(p2, diff);
                        p1 = p1.add(2);
                        p2 = p2.add(2);
                    }
                }
            }
            blk_start += nn;
        }

        // Final normalisation: [0, 4q) → [0, 2q).  n is always a power of two
        // ≥ 2 here, so iterating 512-bit (= 2 coefficients of 4×u64) is safe.
        let q2_512 = pack_512(q2, q2);
        let ptr_512 = begin as *mut __m512i;
        let chunks = n / 2;
        for i in 0..chunks {
            let x = _mm512_loadu_si512(ptr_512.add(i));
            _mm512_storeu_si512(ptr_512.add(i), cond_sub_2q_si512(x, q2_512));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// Butterfly values live in `[0, 4q)`.  The final pointwise Harvey pass reduces
/// to `[0, 2q)` automatically.  Inner levels (≤ `NTT_BLOCK`) are performed
/// block-by-block to keep the working set in cache across all levels.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
pub(crate) unsafe fn intt_avx512<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let q = {
            let a = P::Q[0];
            let b = P::Q[1];
            let c = P::Q[2];
            use core::arch::x86_64::_mm256_set_epi64x;
            _mm256_set_epi64x(0, c as i64, b as i64, a as i64)
        };
        let q4 = _mm256_loadu_si256(table.q4.as_ptr() as *const __m256i);

        let block = NTT_BLOCK.min(n);

        // Precompute segment offsets for inner (block-local) levels: nn = 2, 4, …, block.
        let mut inner_segs = [0usize; 17];
        let mut inner_nn = [0usize; 17];
        let mut num_inner = 0usize;
        let mut seg_avx = 0usize;
        {
            let mut m = 2usize;
            while m <= block {
                inner_nn[num_inner] = m;
                inner_segs[num_inner] = seg_avx;
                let halfm = m / 2;
                if halfm > 1 {
                    seg_avx += 2 * (halfm - 1);
                }
                m *= 2;
                num_inner += 1;
            }
        }

        // Fuse nn=2,4,8 into a single radix-8 pass when available.
        let radix8_start = num_inner >= 3;
        let inner_start = if radix8_start { 3 } else { 0 };

        let mut blk_start = 0usize;
        while blk_start < n {
            let blk_begin = begin.add(blk_start);
            let blk_end = begin.add(blk_start + block) as *const __m256i;

            if radix8_start {
                intt_radix8_first3_ifma(blk_begin, blk_end, q, q4, po_base);
            }

            for i in inner_start..num_inner {
                let m = inner_nn[i];
                let halfm = m / 2;
                let seg = inner_segs[i];
                if halfm > 1 {
                    let count = halfm - 1;
                    intt_iter_ifma(m, blk_begin, blk_end, q, q4, po_base.add(seg), po_base.add(seg + count));
                } else {
                    // m == 2: add/sub only.
                    let mut p1 = blk_begin;
                    let mut p2 = blk_begin.add(1);
                    while (p1 as usize) < (blk_end as usize) {
                        let a = _mm256_loadu_si256(p1);
                        let b = _mm256_loadu_si256(p2);
                        let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                        let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                        _mm256_storeu_si256(p1, sum);
                        _mm256_storeu_si256(p2, diff);
                        p1 = p1.add(2);
                        p2 = p2.add(2);
                    }
                }
            }
            blk_start += block;
        }

        // Outer butterfly levels (breadth-first) for nn > block.
        let mut nn = block * 2;
        while nn <= n {
            let halfnn = nn / 2;
            let count = halfnn - 1;
            intt_iter_ifma(nn, begin, end, q, q4, po_base.add(seg_avx), po_base.add(seg_avx + count));
            seg_avx += 2 * count;
            nn *= 2;
        }

        // Last pass: a[i] *= ω^{-i} / n — Harvey absorbs [0, 4q) → [0, 2q).
        ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt126_ifma::{
        primes::Primes42,
        reference::{
            arithmetic::b_ntt126_ifma_from_znx64_ref,
            ntt::{intt126_ifma_ref, ntt126_ifma_ref},
        },
        tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
    };

    #[test]
    fn harvey_modmul_simd_vs_scalar() {
        use crate::ntt126_ifma::tables::{harvey_modmul, harvey_quotient};

        let q_arr = Primes42::Q;
        for &q in &q_arr {
            let omega = q / 2; // arbitrary twiddle
            let oq = harvey_quotient(omega, q);
            for &a in &[0u64, 1, q - 1, q, 2 * q - 1, q / 3, 42] {
                if a >= 2 * q {
                    continue;
                }

                let expected = harvey_modmul(a, omega, oq, q);

                // SIMD version: pack into lane 0
                let a_vec = [a as i64, 0i64, 0, 0];
                let o_vec = [omega as i64, 0i64, 0, 0];
                let oq_vec = [oq as i64, 0i64, 0, 0];
                let q_vec = [q as i64, 0i64, 0, 0];

                let got = unsafe {
                    let av = _mm256_loadu_si256(a_vec.as_ptr() as *const __m256i);
                    let ov = _mm256_loadu_si256(o_vec.as_ptr() as *const __m256i);
                    let oqv = _mm256_loadu_si256(oq_vec.as_ptr() as *const __m256i);
                    let qv = _mm256_loadu_si256(q_vec.as_ptr() as *const __m256i);
                    let r = harvey_modmul_si256(av, ov, oqv, qv);
                    let mut out = [0i64; 4];
                    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, r);
                    out[0] as u64
                };

                assert_eq!(
                    got % q,
                    expected % q,
                    "SIMD harvey_modmul mismatch: a={a}, omega={omega}, q={q}, got={got}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn ntt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt126IfmaTable::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
            b_ntt126_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx512::<Primes42>(&fwd, &mut data_avx) };
            ntt126_ifma_ref::<Primes42>(&fwd, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(
                    data_avx[i], data_ref[i],
                    "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i], data_ref[i]
                );
            }
        }
    }

    #[test]
    fn intt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt126IfmaTable::<Primes42>::new(n);
            let inv = Ntt126IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 4 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data, &coeffs);
            ntt126_ifma_ref::<Primes42>(&fwd, &mut data);

            let mut data_avx = data.clone();
            let mut data_ref = data.clone();

            unsafe { intt_avx512::<Primes42>(&inv, &mut data_avx) };
            intt126_ifma_ref::<Primes42>(&inv, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(
                    data_avx[i], data_ref[i],
                    "n={n} idx={i}: iNTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i], data_ref[i]
                );
            }
        }
    }

    #[test]
    fn ntt_intt_avx512_roundtrip() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt126IfmaTable::<Primes42>::new(n);
            let inv = Ntt126IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 4 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let orig = data.clone();

            unsafe {
                ntt_avx512::<Primes42>(&fwd, &mut data);
                intt_avx512::<Primes42>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..3 {
                    let o = orig[4 * i + k] % Primes42::Q[k];
                    let g = data[4 * i + k] % Primes42::Q[k];
                    assert_eq!(o, g, "n={n} i={i} k={k}: roundtrip mismatch");
                }
            }
        }
    }
}
