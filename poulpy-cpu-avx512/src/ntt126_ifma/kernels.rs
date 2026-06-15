//! Raw AVX512-IFMA forward and inverse NTT kernels.
//!
//! These kernels are the core arithmetic engine of the IFMA backend.
//!
//! - Butterfly values live in `[0, 4q)`; a single final pass renormalises to `[0, 2q)`.
//! - Diff path feeds directly into Harvey without a pre-reduction (IFMA's 52-bit
//!   product absorbs inputs up to `2^52`).
//! - Harvey multiplication replaces the AVX2 split-precomputed multiply path.
//! - Two coefficients are processed at a time through 512-bit loads where profitable.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64,
    _mm256_mask_blend_epi64, _mm256_min_epu64, _mm256_set_epi64x, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_storeu_si256,
    _mm256_sub_epi64, _mm512_add_epi64, _mm512_and_si512, _mm512_castsi256_si512, _mm512_inserti64x4, _mm512_loadu_si512,
    _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_storeu_epi64, _mm512_maskz_loadu_epi64, _mm512_min_epu64,
    _mm512_set1_epi64, _mm512_setzero_si512, _mm512_shuffle_i64x2, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_unpackhi_epi64,
    _mm512_unpacklo_epi64,
};

use std::mem::size_of;

use crate::ntt126_ifma::{
    primes::PrimeSetNtt126Ifma,
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv, cond_sub_2q, harvey_modmul},
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

/// Fused level-0 twist and first butterfly level: for each pair
/// `(i, i + n/2)`, twist both elements by their level-0 omegas, then apply
/// the level-1 butterfly, saving one full store/reload pass over the data.
/// Operations per element match the unfused sequence exactly.
#[target_feature(enable = "avx512ifma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ntt_iter_first_fused_ifma(
    begin: *mut __m256i,
    halfn: usize,
    po0_omega: *const __m256i,
    po0_quot: *const __m256i,
    po1_omega: *const __m256i,
    po1_quot: *const __m256i,
    q: __m256i,
    q4: __m256i,
) {
    unsafe {
        let mut ptr1 = begin;
        let mut ptr2 = begin.add(halfn);

        // i = 0: twist both halves, butterfly without a level-1 twiddle.
        {
            let a = harvey_modmul_si256(
                _mm256_loadu_si256(ptr1),
                _mm256_loadu_si256(po0_omega),
                _mm256_loadu_si256(po0_quot),
                q,
            );
            let b = harvey_modmul_si256(
                _mm256_loadu_si256(ptr2),
                _mm256_loadu_si256(po0_omega.add(halfn)),
                _mm256_loadu_si256(po0_quot.add(halfn)),
                q,
            );
            let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
            let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
            _mm256_storeu_si256(ptr1, sum);
            _mm256_storeu_si256(ptr2, diff);
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);
        }

        // i = 1: peel so the 512-bit loop starts on even indices.
        {
            let a = harvey_modmul_si256(
                _mm256_loadu_si256(ptr1),
                _mm256_loadu_si256(po0_omega.add(1)),
                _mm256_loadu_si256(po0_quot.add(1)),
                q,
            );
            let b = harvey_modmul_si256(
                _mm256_loadu_si256(ptr2),
                _mm256_loadu_si256(po0_omega.add(halfn + 1)),
                _mm256_loadu_si256(po0_quot.add(halfn + 1)),
                q,
            );
            let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
            let diff = _mm256_sub_epi64(_mm256_add_epi64(a, q4), b);
            _mm256_storeu_si256(ptr1, sum);
            _mm256_storeu_si256(
                ptr2,
                harvey_modmul_si256(diff, _mm256_loadu_si256(po1_omega), _mm256_loadu_si256(po1_quot), q),
            );
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);
        }

        // i = 2..halfn: 512-bit pairs.
        let q_512 = pack_512(q, q);
        let q4_512 = pack_512(q4, q4);
        let pairs = (halfn - 2) / 2;
        let tw0a_512 = po0_omega.add(2) as *const __m512i;
        let tw0aq_512 = po0_quot.add(2) as *const __m512i;
        let tw0b_512 = po0_omega.add(halfn + 2) as *const __m512i;
        let tw0bq_512 = po0_quot.add(halfn + 2) as *const __m512i;
        let tw1_512 = po1_omega.add(1) as *const __m512i;
        let tw1q_512 = po1_quot.add(1) as *const __m512i;
        for p in 0..pairs {
            let a = harvey_modmul_si512(
                _mm512_loadu_si512(ptr1 as *const __m512i),
                _mm512_loadu_si512(tw0a_512.add(p)),
                _mm512_loadu_si512(tw0aq_512.add(p)),
                q_512,
            );
            let b = harvey_modmul_si512(
                _mm512_loadu_si512(ptr2 as *const __m512i),
                _mm512_loadu_si512(tw0b_512.add(p)),
                _mm512_loadu_si512(tw0bq_512.add(p)),
                q_512,
            );
            let sum = cond_sub_2q_si512(_mm512_add_epi64(a, b), q4_512);
            let diff = _mm512_sub_epi64(_mm512_add_epi64(a, q4_512), b);
            _mm512_storeu_si512(ptr1 as *mut __m512i, sum);
            _mm512_storeu_si512(
                ptr2 as *mut __m512i,
                harvey_modmul_si512(
                    diff,
                    _mm512_loadu_si512(tw1_512.add(p)),
                    _mm512_loadu_si512(tw1q_512.add(p)),
                    q_512,
                ),
            );
            ptr1 = ptr1.add(2);
            ptr2 = ptr2.add(2);
        }

        // 256-bit tail.
        for i in (2 + 2 * pairs)..halfn {
            let a = harvey_modmul_si256(
                _mm256_loadu_si256(ptr1),
                _mm256_loadu_si256(po0_omega.add(i)),
                _mm256_loadu_si256(po0_quot.add(i)),
                q,
            );
            let b = harvey_modmul_si256(
                _mm256_loadu_si256(ptr2),
                _mm256_loadu_si256(po0_omega.add(halfn + i)),
                _mm256_loadu_si256(po0_quot.add(halfn + i)),
                q,
            );
            let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
            let diff = _mm256_sub_epi64(_mm256_add_epi64(a, q4), b);
            _mm256_storeu_si256(ptr1, sum);
            _mm256_storeu_si256(
                ptr2,
                harvey_modmul_si256(
                    diff,
                    _mm256_loadu_si256(po1_omega.add(i - 1)),
                    _mm256_loadu_si256(po1_quot.add(i - 1)),
                    q,
                ),
            );
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);
        }
    }
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

/// Forward mirror of [`intt_radix8_first3_ifma`]: fuses the last three
/// butterfly levels (`nn = 8, 4, 2`) of each block in one register pass.
/// Twiddles are block-independent; operations per element match the unfused
/// per-level sequence exactly.
#[target_feature(enable = "avx512ifma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ntt_radix8_last3_ifma(
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q4: __m256i,
    w8: *const __m256i,
    w8q: *const __m256i,
    w4: *const __m256i,
    w4q: *const __m256i,
) {
    unsafe {
        let w8_1 = _mm256_loadu_si256(w8);
        let w8_2 = _mm256_loadu_si256(w8.add(1));
        let w8_3 = _mm256_loadu_si256(w8.add(2));
        let w8_1q = _mm256_loadu_si256(w8q);
        let w8_2q = _mm256_loadu_si256(w8q.add(1));
        let w8_3q = _mm256_loadu_si256(w8q.add(2));
        let w4_1 = _mm256_loadu_si256(w4);
        let w4_1q = _mm256_loadu_si256(w4q);

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

            // nn=8: identity at i=0, twiddles w8_{1,2,3} at i=1..3.
            let u0 = cond_sub_2q_si256(_mm256_add_epi64(a0, a4), q4);
            let u4 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a0, q4), a4), q4);
            let u1 = cond_sub_2q_si256(_mm256_add_epi64(a1, a5), q4);
            let u5 = harvey_modmul_si256(_mm256_sub_epi64(_mm256_add_epi64(a1, q4), a5), w8_1, w8_1q, q);
            let u2 = cond_sub_2q_si256(_mm256_add_epi64(a2, a6), q4);
            let u6 = harvey_modmul_si256(_mm256_sub_epi64(_mm256_add_epi64(a2, q4), a6), w8_2, w8_2q, q);
            let u3 = cond_sub_2q_si256(_mm256_add_epi64(a3, a7), q4);
            let u7 = harvey_modmul_si256(_mm256_sub_epi64(_mm256_add_epi64(a3, q4), a7), w8_3, w8_3q, q);

            // nn=4: identity at i=0, twiddle w4 at i=1, per half-block.
            let v0 = cond_sub_2q_si256(_mm256_add_epi64(u0, u2), q4);
            let v2 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u0, q4), u2), q4);
            let v1 = cond_sub_2q_si256(_mm256_add_epi64(u1, u3), q4);
            let v3 = harvey_modmul_si256(_mm256_sub_epi64(_mm256_add_epi64(u1, q4), u3), w4_1, w4_1q, q);
            let v4 = cond_sub_2q_si256(_mm256_add_epi64(u4, u6), q4);
            let v6 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(u4, q4), u6), q4);
            let v5 = cond_sub_2q_si256(_mm256_add_epi64(u5, u7), q4);
            let v7 = harvey_modmul_si256(_mm256_sub_epi64(_mm256_add_epi64(u5, q4), u7), w4_1, w4_1q, q);

            // nn=2: 4 identity butterflies.
            let o0 = cond_sub_2q_si256(_mm256_add_epi64(v0, v1), q4);
            let o1 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(v0, q4), v1), q4);
            let o2 = cond_sub_2q_si256(_mm256_add_epi64(v2, v3), q4);
            let o3 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(v2, q4), v3), q4);
            let o4 = cond_sub_2q_si256(_mm256_add_epi64(v4, v5), q4);
            let o5 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(v4, q4), v5), q4);
            let o6 = cond_sub_2q_si256(_mm256_add_epi64(v6, v7), q4);
            let o7 = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(v6, q4), v7), q4);

            _mm256_storeu_si256(ptr, o0);
            _mm256_storeu_si256(ptr.add(1), o1);
            _mm256_storeu_si256(ptr.add(2), o2);
            _mm256_storeu_si256(ptr.add(3), o3);
            _mm256_storeu_si256(ptr.add(4), o4);
            _mm256_storeu_si256(ptr.add(5), o5);
            _mm256_storeu_si256(ptr.add(6), o6);
            _mm256_storeu_si256(ptr.add(7), o7);

            ptr = ptr.add(8);
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

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load_plane_twiddles8(powomega: &[u64], base: usize, count: usize, idx: usize, prime: usize) -> __m512i {
    unsafe { _mm512_loadu_si512(powomega.as_ptr().add(base + prime * count + idx) as *const __m512i) }
}

/// Masked (zeroing, fault-suppressing) twiddle load for sub-8 remainders.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn maskz_load_plane_twiddles8(mask: u8, powomega: &[u64], base: usize, count: usize, idx: usize, prime: usize) -> __m512i {
    unsafe { _mm512_maskz_loadu_epi64(mask, powomega.as_ptr().add(base + prime * count + idx) as *const i64) }
}

#[target_feature(enable = "avx512ifma")]
#[inline]
unsafe fn harvey_modmul_plane8(a: __m512i, omega: __m512i, omega_quot: __m512i, q: u64) -> __m512i {
    unsafe { harvey_modmul_si512(a, omega, omega_quot, _mm512_set1_epi64(q as i64)) }
}

/// In-register 8×8 transpose of u64 lanes across eight `__m512i` rows.
/// Row `k` holds 8 consecutive coefficients of block `k`; after the transpose
/// lane vector `j` holds coefficient position `j` across the eight blocks, so
/// the radix-8 butterflies become permute-free vertical ops with broadcast
/// twiddles. The transpose is its own inverse.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn transpose8x8_epi64(r: [__m512i; 8]) -> [__m512i; 8] {
    // Stage 1: interleave 64-bit lanes within 128-bit groups.
    let a0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi64(r[6], r[7]);
    // Stage 2: interleave 128-bit chunks (0x88 picks chunks 0,2; 0xDD picks 1,3).
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b2 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b5 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b6 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    // Stage 3: interleave 256-bit halves.
    [
        _mm512_shuffle_i64x2::<0x88>(b0, b4),
        _mm512_shuffle_i64x2::<0x88>(b1, b5),
        _mm512_shuffle_i64x2::<0x88>(b2, b6),
        _mm512_shuffle_i64x2::<0x88>(b3, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b4),
        _mm512_shuffle_i64x2::<0xDD>(b1, b5),
        _mm512_shuffle_i64x2::<0xDD>(b2, b6),
        _mm512_shuffle_i64x2::<0xDD>(b3, b7),
    ]
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn ntt_plane_radix8_last3(plane: &mut [u64], seg_base: usize, prime: usize, q: u64, q4: u64, powomega: &[u64]) {
    let mut u0 = [0u64; 4];
    let mut u4 = [0u64; 4];
    let ptr = plane.as_mut_ptr();

    unsafe {
        let qv = _mm256_set1_epi64x(q as i64);
        let q4v = _mm256_set1_epi64x(q4 as i64);

        let w8_base = seg_base;
        let w8_quot_base = seg_base + 3 * 3;
        let w8 = _mm256_set_epi64x(
            powomega[w8_base + prime * 3 + 2] as i64,
            powomega[w8_base + prime * 3 + 1] as i64,
            powomega[w8_base + prime * 3] as i64,
            1,
        );
        let w8q = _mm256_set_epi64x(
            powomega[w8_quot_base + prime * 3 + 2] as i64,
            powomega[w8_quot_base + prime * 3 + 1] as i64,
            powomega[w8_quot_base + prime * 3] as i64,
            0,
        );

        let w4_base = seg_base + 6 * 3;
        let w4_quot_base = w4_base + 3;
        let w4 = powomega[w4_base + prime];
        let w4q = powomega[w4_quot_base + prime];

        // Vectorized path: process 8 blocks (64 coefficients) per iteration.
        // Transpose so each zmm holds one coefficient position across the 8
        // blocks; the radix-8 butterflies are then permute-free vertical ops
        // with broadcast twiddles.
        let q4w = _mm512_set1_epi64(q4 as i64);
        let q2w = _mm512_set1_epi64((q4 >> 1) as i64);
        let w8_1 = _mm512_set1_epi64(powomega[w8_base + prime * 3] as i64);
        let w8_2 = _mm512_set1_epi64(powomega[w8_base + prime * 3 + 1] as i64);
        let w8_3 = _mm512_set1_epi64(powomega[w8_base + prime * 3 + 2] as i64);
        let w8q_1 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3] as i64);
        let w8q_2 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3 + 1] as i64);
        let w8q_3 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3 + 2] as i64);
        let w4v = _mm512_set1_epi64(w4 as i64);
        let w4qv = _mm512_set1_epi64(w4q as i64);

        let ngroups = plane.len() / 64;
        for g in 0..ngroups {
            let base = g * 64;
            let c = transpose8x8_epi64([
                _mm512_loadu_si512(ptr.add(base) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 8) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 16) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 24) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 32) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 40) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 48) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 56) as *const __m512i),
            ]);

            // nn = 8: pairs (j, j+4), twiddles [1, w8_1, w8_2, w8_3].
            let s0 = cond_sub_2q_si512(_mm512_add_epi64(c[0], c[4]), q4w);
            let d0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(c[0], q4w), c[4]), q4w);
            let s1 = cond_sub_2q_si512(_mm512_add_epi64(c[1], c[5]), q4w);
            let d1 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(c[1], q4w), c[5]), w8_1, w8q_1, q);
            let s2 = cond_sub_2q_si512(_mm512_add_epi64(c[2], c[6]), q4w);
            let d2 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(c[2], q4w), c[6]), w8_2, w8q_2, q);
            let s3 = cond_sub_2q_si512(_mm512_add_epi64(c[3], c[7]), q4w);
            let d3 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(c[3], q4w), c[7]), w8_3, w8q_3, q);

            // nn = 4: pairs (s0,s2),(s1,s3),(d0,d2),(d1,d3), twiddles [1, w4].
            let v0 = cond_sub_2q_si512(_mm512_add_epi64(s0, s2), q4w);
            let v2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(s0, q4w), s2), q4w);
            let v1 = cond_sub_2q_si512(_mm512_add_epi64(s1, s3), q4w);
            let v3 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(s1, q4w), s3), w4v, w4qv, q);
            let v4 = cond_sub_2q_si512(_mm512_add_epi64(d0, d2), q4w);
            let v6 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(d0, q4w), d2), q4w);
            let v5 = cond_sub_2q_si512(_mm512_add_epi64(d1, d3), q4w);
            let v7 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(d1, q4w), d3), w4v, w4qv, q);

            // nn = 2: pairs (v0,v1),(v2,v3),(v4,v5),(v6,v7), twiddle 1. The final
            // [0,4q) -> [0,2q) normalization is folded in here (q2w), removing a
            // separate full pass over the plane.
            let o0 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_add_epi64(v0, v1), q4w), q2w);
            let o1 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(v0, q4w), v1), q4w), q2w);
            let o2 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_add_epi64(v2, v3), q4w), q2w);
            let o3 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(v2, q4w), v3), q4w), q2w);
            let o4 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_add_epi64(v4, v5), q4w), q2w);
            let o5 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(v4, q4w), v5), q4w), q2w);
            let o6 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_add_epi64(v6, v7), q4w), q2w);
            let o7 = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(v6, q4w), v7), q4w), q2w);

            let out = transpose8x8_epi64([o0, o1, o2, o3, o4, o5, o6, o7]);
            _mm512_storeu_si512(ptr.add(base) as *mut __m512i, out[0]);
            _mm512_storeu_si512(ptr.add(base + 8) as *mut __m512i, out[1]);
            _mm512_storeu_si512(ptr.add(base + 16) as *mut __m512i, out[2]);
            _mm512_storeu_si512(ptr.add(base + 24) as *mut __m512i, out[3]);
            _mm512_storeu_si512(ptr.add(base + 32) as *mut __m512i, out[4]);
            _mm512_storeu_si512(ptr.add(base + 40) as *mut __m512i, out[5]);
            _mm512_storeu_si512(ptr.add(base + 48) as *mut __m512i, out[6]);
            _mm512_storeu_si512(ptr.add(base + 56) as *mut __m512i, out[7]);
        }

        let mut block = ngroups * 64;
        while block < plane.len() {
            let lo = _mm256_loadu_si256(ptr.add(block) as *const __m256i);
            let hi = _mm256_loadu_si256(ptr.add(block + 4) as *const __m256i);

            let sum = cond_sub_2q_si256(_mm256_add_epi64(lo, hi), q4v);
            let diff = _mm256_sub_epi64(_mm256_add_epi64(lo, q4v), hi);
            let diff_id = cond_sub_2q_si256(diff, q4v);
            let diff_mul = harvey_modmul_si256(diff, w8, w8q, qv);
            let out_hi = _mm256_mask_blend_epi64(0b1110, diff_id, diff_mul);

            _mm256_storeu_si256(u0.as_mut_ptr() as *mut __m256i, sum);
            _mm256_storeu_si256(u4.as_mut_ptr() as *mut __m256i, out_hi);

            let v0 = cond_sub_2q(u0[0] + u0[2], q4);
            let v2 = cond_sub_2q(u0[0] + q4 - u0[2], q4);
            let v1 = cond_sub_2q(u0[1] + u0[3], q4);
            let v3 = harvey_modmul(u0[1] + q4 - u0[3], w4, w4q, q);
            let v4 = cond_sub_2q(u4[0] + u4[2], q4);
            let v6 = cond_sub_2q(u4[0] + q4 - u4[2], q4);
            let v5 = cond_sub_2q(u4[1] + u4[3], q4);
            let v7 = harvey_modmul(u4[1] + q4 - u4[3], w4, w4q, q);

            let q2 = q4 >> 1;
            plane[block] = cond_sub_2q(cond_sub_2q(v0 + v1, q4), q2);
            plane[block + 1] = cond_sub_2q(cond_sub_2q(v0 + q4 - v1, q4), q2);
            plane[block + 2] = cond_sub_2q(cond_sub_2q(v2 + v3, q4), q2);
            plane[block + 3] = cond_sub_2q(cond_sub_2q(v2 + q4 - v3, q4), q2);
            plane[block + 4] = cond_sub_2q(cond_sub_2q(v4 + v5, q4), q2);
            plane[block + 5] = cond_sub_2q(cond_sub_2q(v4 + q4 - v5, q4), q2);
            plane[block + 6] = cond_sub_2q(cond_sub_2q(v6 + v7, q4), q2);
            plane[block + 7] = cond_sub_2q(cond_sub_2q(v6 + q4 - v7, q4), q2);

            block += 8;
        }
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_plane_radix8_first3(plane: &mut [u64], seg_base: usize, prime: usize, q: u64, q4: u64, powomega: &[u64]) {
    let mut lo_lanes = [0u64; 4];
    let mut hi_lanes = [0u64; 4];
    let mut out = [0u64; 8];

    unsafe {
        let qv = _mm256_set1_epi64x(q as i64);
        let q4v = _mm256_set1_epi64x(q4 as i64);

        let w4_base = seg_base;
        let w4_quot_base = seg_base + 3;
        let w4 = powomega[w4_base + prime];
        let w4q = powomega[w4_quot_base + prime];

        let w8_base = seg_base + 6;
        let w8_quot_base = w8_base + 3 * 3;
        let w8 = _mm256_set_epi64x(
            powomega[w8_base + prime * 3 + 2] as i64,
            powomega[w8_base + prime * 3 + 1] as i64,
            powomega[w8_base + prime * 3] as i64,
            1,
        );
        let w8q = _mm256_set_epi64x(
            powomega[w8_quot_base + prime * 3 + 2] as i64,
            powomega[w8_quot_base + prime * 3 + 1] as i64,
            powomega[w8_quot_base + prime * 3] as i64,
            0,
        );

        // Vectorized path: 8 blocks (64 coefficients) per iteration, transposed
        // so each zmm is one coefficient position across the 8 blocks.
        let q4w = _mm512_set1_epi64(q4 as i64);
        let w4v = _mm512_set1_epi64(w4 as i64);
        let w4qv = _mm512_set1_epi64(w4q as i64);
        let w8_1 = _mm512_set1_epi64(powomega[w8_base + prime * 3] as i64);
        let w8_2 = _mm512_set1_epi64(powomega[w8_base + prime * 3 + 1] as i64);
        let w8_3 = _mm512_set1_epi64(powomega[w8_base + prime * 3 + 2] as i64);
        let w8q_1 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3] as i64);
        let w8q_2 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3 + 1] as i64);
        let w8q_3 = _mm512_set1_epi64(powomega[w8_quot_base + prime * 3 + 2] as i64);

        let ptr = plane.as_mut_ptr();
        let ngroups = plane.len() / 64;
        for g in 0..ngroups {
            let base = g * 64;
            let c = transpose8x8_epi64([
                _mm512_loadu_si512(ptr.add(base) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 8) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 16) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 24) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 32) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 40) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 48) as *const __m512i),
                _mm512_loadu_si512(ptr.add(base + 56) as *const __m512i),
            ]);

            // nn = 2: pairs (j, j+1), twiddle 1.
            let t0 = cond_sub_2q_si512(_mm512_add_epi64(c[0], c[1]), q4w);
            let t1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(c[0], q4w), c[1]), q4w);
            let t2 = cond_sub_2q_si512(_mm512_add_epi64(c[2], c[3]), q4w);
            let t3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(c[2], q4w), c[3]), q4w);
            let t4 = cond_sub_2q_si512(_mm512_add_epi64(c[4], c[5]), q4w);
            let t5 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(c[4], q4w), c[5]), q4w);
            let t6 = cond_sub_2q_si512(_mm512_add_epi64(c[6], c[7]), q4w);
            let t7 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(c[6], q4w), c[7]), q4w);

            // nn = 4: twiddle w4 applied to t3, t7 before combining.
            let u0 = cond_sub_2q_si512(_mm512_add_epi64(t0, t2), q4w);
            let u2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(t0, q4w), t2), q4w);
            let bo13 = harvey_modmul_plane8(t3, w4v, w4qv, q);
            let u1 = cond_sub_2q_si512(_mm512_add_epi64(t1, bo13), q4w);
            let u3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(t1, q4w), bo13), q4w);
            let u4 = cond_sub_2q_si512(_mm512_add_epi64(t4, t6), q4w);
            let u6 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(t4, q4w), t6), q4w);
            let bo57 = harvey_modmul_plane8(t7, w4v, w4qv, q);
            let u5 = cond_sub_2q_si512(_mm512_add_epi64(t5, bo57), q4w);
            let u7 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(t5, q4w), bo57), q4w);

            // nn = 8: pairs (j, j+4), twiddles [1, w8_1, w8_2, w8_3] on the high half.
            let bo0 = u4;
            let bo1 = harvey_modmul_plane8(u5, w8_1, w8q_1, q);
            let bo2 = harvey_modmul_plane8(u6, w8_2, w8q_2, q);
            let bo3 = harvey_modmul_plane8(u7, w8_3, w8q_3, q);
            let o0 = cond_sub_2q_si512(_mm512_add_epi64(u0, bo0), q4w);
            let o4 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u0, q4w), bo0), q4w);
            let o1 = cond_sub_2q_si512(_mm512_add_epi64(u1, bo1), q4w);
            let o5 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u1, q4w), bo1), q4w);
            let o2 = cond_sub_2q_si512(_mm512_add_epi64(u2, bo2), q4w);
            let o6 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u2, q4w), bo2), q4w);
            let o3 = cond_sub_2q_si512(_mm512_add_epi64(u3, bo3), q4w);
            let o7 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u3, q4w), bo3), q4w);

            let outv = transpose8x8_epi64([o0, o1, o2, o3, o4, o5, o6, o7]);
            _mm512_storeu_si512(ptr.add(base) as *mut __m512i, outv[0]);
            _mm512_storeu_si512(ptr.add(base + 8) as *mut __m512i, outv[1]);
            _mm512_storeu_si512(ptr.add(base + 16) as *mut __m512i, outv[2]);
            _mm512_storeu_si512(ptr.add(base + 24) as *mut __m512i, outv[3]);
            _mm512_storeu_si512(ptr.add(base + 32) as *mut __m512i, outv[4]);
            _mm512_storeu_si512(ptr.add(base + 40) as *mut __m512i, outv[5]);
            _mm512_storeu_si512(ptr.add(base + 48) as *mut __m512i, outv[6]);
            _mm512_storeu_si512(ptr.add(base + 56) as *mut __m512i, outv[7]);
        }

        let mut block = ngroups * 64;
        while block < plane.len() {
            let a0 = plane[block];
            let a1 = plane[block + 1];
            let a2 = plane[block + 2];
            let a3 = plane[block + 3];
            let a4 = plane[block + 4];
            let a5 = plane[block + 5];
            let a6 = plane[block + 6];
            let a7 = plane[block + 7];

            let t0 = cond_sub_2q(a0 + a1, q4);
            let t1 = cond_sub_2q(a0 + q4 - a1, q4);
            let t2 = cond_sub_2q(a2 + a3, q4);
            let t3 = cond_sub_2q(a2 + q4 - a3, q4);
            let t4 = cond_sub_2q(a4 + a5, q4);
            let t5 = cond_sub_2q(a4 + q4 - a5, q4);
            let t6 = cond_sub_2q(a6 + a7, q4);
            let t7 = cond_sub_2q(a6 + q4 - a7, q4);

            let u0 = cond_sub_2q(t0 + t2, q4);
            let u2 = cond_sub_2q(t0 + q4 - t2, q4);
            let bo13 = harvey_modmul(t3, w4, w4q, q);
            let u1 = cond_sub_2q(t1 + bo13, q4);
            let u3 = cond_sub_2q(t1 + q4 - bo13, q4);

            let u4 = cond_sub_2q(t4 + t6, q4);
            let u6 = cond_sub_2q(t4 + q4 - t6, q4);
            let bo57 = harvey_modmul(t7, w4, w4q, q);
            let u5 = cond_sub_2q(t5 + bo57, q4);
            let u7 = cond_sub_2q(t5 + q4 - bo57, q4);

            lo_lanes[0] = u0;
            lo_lanes[1] = u1;
            lo_lanes[2] = u2;
            lo_lanes[3] = u3;
            hi_lanes[0] = u4;
            hi_lanes[1] = u5;
            hi_lanes[2] = u6;
            hi_lanes[3] = u7;

            let lo = _mm256_loadu_si256(lo_lanes.as_ptr() as *const __m256i);
            let hi = _mm256_loadu_si256(hi_lanes.as_ptr() as *const __m256i);
            let bo_mul = harvey_modmul_si256(hi, w8, w8q, qv);
            let bo = _mm256_mask_blend_epi64(0b1110, hi, bo_mul);
            let out_lo = cond_sub_2q_si256(_mm256_add_epi64(lo, bo), q4v);
            let out_hi = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(lo, q4v), bo), q4v);

            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, out_lo);
            _mm256_storeu_si256(out.as_mut_ptr().add(4) as *mut __m256i, out_hi);
            plane[block] = out[0];
            plane[block + 1] = out[1];
            plane[block + 2] = out[2];
            plane[block + 3] = out[3];
            plane[block + 4] = out[4];
            plane[block + 5] = out[5];
            plane[block + 6] = out[6];
            plane[block + 7] = out[7];

            block += 8;
        }
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn ntt_plane_first_fused(plane: &mut [u64], prime: usize, q: u64, q4: u64, q4v: __m512i, powomega: &[u64]) {
    let n = plane.len();
    let halfn = n / 2;
    let count = halfn - 1;
    let ptr = plane.as_mut_ptr();

    unsafe {
        let omega0_base = 0usize;
        let quot0_base = 3 * n;
        let omega1_base = 6 * n;
        let quot1_base = omega1_base + 3 * count;

        {
            let a = harvey_modmul(
                plane[0],
                powomega[omega0_base + prime * n],
                powomega[quot0_base + prime * n],
                q,
            );
            let b = harvey_modmul(
                plane[halfn],
                powomega[omega0_base + prime * n + halfn],
                powomega[quot0_base + prime * n + halfn],
                q,
            );
            plane[0] = cond_sub_2q(a + b, q4);
            plane[halfn] = cond_sub_2q(a + q4 - b, q4);
        }

        let mut i = 1usize;
        while i + 32 <= halfn {
            let p1 = i;
            let p2 = halfn + i;

            let a0 = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
            let b0 = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
            let a1 = _mm512_loadu_si512(ptr.add(p1 + 8) as *const __m512i);
            let b1 = _mm512_loadu_si512(ptr.add(p2 + 8) as *const __m512i);
            let a2 = _mm512_loadu_si512(ptr.add(p1 + 16) as *const __m512i);
            let b2 = _mm512_loadu_si512(ptr.add(p2 + 16) as *const __m512i);
            let a3 = _mm512_loadu_si512(ptr.add(p1 + 24) as *const __m512i);
            let b3 = _mm512_loadu_si512(ptr.add(p2 + 24) as *const __m512i);

            let a0 = harvey_modmul_plane8(
                a0,
                load_plane_twiddles8(powomega, omega0_base, n, p1, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p1, prime),
                q,
            );
            let b0 = harvey_modmul_plane8(
                b0,
                load_plane_twiddles8(powomega, omega0_base, n, p2, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p2, prime),
                q,
            );
            let a1 = harvey_modmul_plane8(
                a1,
                load_plane_twiddles8(powomega, omega0_base, n, p1 + 8, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p1 + 8, prime),
                q,
            );
            let b1 = harvey_modmul_plane8(
                b1,
                load_plane_twiddles8(powomega, omega0_base, n, p2 + 8, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p2 + 8, prime),
                q,
            );
            let a2 = harvey_modmul_plane8(
                a2,
                load_plane_twiddles8(powomega, omega0_base, n, p1 + 16, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p1 + 16, prime),
                q,
            );
            let b2 = harvey_modmul_plane8(
                b2,
                load_plane_twiddles8(powomega, omega0_base, n, p2 + 16, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p2 + 16, prime),
                q,
            );
            let a3 = harvey_modmul_plane8(
                a3,
                load_plane_twiddles8(powomega, omega0_base, n, p1 + 24, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p1 + 24, prime),
                q,
            );
            let b3 = harvey_modmul_plane8(
                b3,
                load_plane_twiddles8(powomega, omega0_base, n, p2 + 24, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p2 + 24, prime),
                q,
            );

            let sum0 = cond_sub_2q_si512(_mm512_add_epi64(a0, b0), q4v);
            let sum1 = cond_sub_2q_si512(_mm512_add_epi64(a1, b1), q4v);
            let sum2 = cond_sub_2q_si512(_mm512_add_epi64(a2, b2), q4v);
            let sum3 = cond_sub_2q_si512(_mm512_add_epi64(a3, b3), q4v);

            let diff0 = _mm512_sub_epi64(_mm512_add_epi64(a0, q4v), b0);
            let diff1 = _mm512_sub_epi64(_mm512_add_epi64(a1, q4v), b1);
            let diff2 = _mm512_sub_epi64(_mm512_add_epi64(a2, q4v), b2);
            let diff3 = _mm512_sub_epi64(_mm512_add_epi64(a3, q4v), b3);

            let out0 = harvey_modmul_plane8(
                diff0,
                load_plane_twiddles8(powomega, omega1_base, count, i - 1, prime),
                load_plane_twiddles8(powomega, quot1_base, count, i - 1, prime),
                q,
            );
            let out1 = harvey_modmul_plane8(
                diff1,
                load_plane_twiddles8(powomega, omega1_base, count, i + 7, prime),
                load_plane_twiddles8(powomega, quot1_base, count, i + 7, prime),
                q,
            );
            let out2 = harvey_modmul_plane8(
                diff2,
                load_plane_twiddles8(powomega, omega1_base, count, i + 15, prime),
                load_plane_twiddles8(powomega, quot1_base, count, i + 15, prime),
                q,
            );
            let out3 = harvey_modmul_plane8(
                diff3,
                load_plane_twiddles8(powomega, omega1_base, count, i + 23, prime),
                load_plane_twiddles8(powomega, quot1_base, count, i + 23, prime),
                q,
            );

            _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum0);
            _mm512_storeu_si512(ptr.add(p1 + 8) as *mut __m512i, sum1);
            _mm512_storeu_si512(ptr.add(p1 + 16) as *mut __m512i, sum2);
            _mm512_storeu_si512(ptr.add(p1 + 24) as *mut __m512i, sum3);
            _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out0);
            _mm512_storeu_si512(ptr.add(p2 + 8) as *mut __m512i, out1);
            _mm512_storeu_si512(ptr.add(p2 + 16) as *mut __m512i, out2);
            _mm512_storeu_si512(ptr.add(p2 + 24) as *mut __m512i, out3);

            i += 32;
        }

        while i + 8 <= halfn {
            let p1 = i;
            let p2 = halfn + i;
            let a = harvey_modmul_plane8(
                _mm512_loadu_si512(ptr.add(p1) as *const __m512i),
                load_plane_twiddles8(powomega, omega0_base, n, p1, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p1, prime),
                q,
            );
            let b = harvey_modmul_plane8(
                _mm512_loadu_si512(ptr.add(p2) as *const __m512i),
                load_plane_twiddles8(powomega, omega0_base, n, p2, prime),
                load_plane_twiddles8(powomega, quot0_base, n, p2, prime),
                q,
            );
            let sum = cond_sub_2q_si512(_mm512_add_epi64(a, b), q4v);
            let diff = _mm512_sub_epi64(_mm512_add_epi64(a, q4v), b);
            let out = harvey_modmul_plane8(
                diff,
                load_plane_twiddles8(powomega, omega1_base, count, i - 1, prime),
                load_plane_twiddles8(powomega, quot1_base, count, i - 1, prime),
                q,
            );
            _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum);
            _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out);
            i += 8;
        }

        while i < halfn {
            let p1 = i;
            let p2 = halfn + i;
            let a = harvey_modmul(
                plane[p1],
                powomega[omega0_base + prime * n + p1],
                powomega[quot0_base + prime * n + p1],
                q,
            );
            let b = harvey_modmul(
                plane[p2],
                powomega[omega0_base + prime * n + p2],
                powomega[quot0_base + prime * n + p2],
                q,
            );
            plane[p1] = cond_sub_2q(a + b, q4);
            plane[p2] = harvey_modmul(
                a + q4 - b,
                powomega[omega1_base + prime * count + i - 1],
                powomega[quot1_base + prime * count + i - 1],
                q,
            );
            i += 1;
        }
    }
}

const NTT_BLOCK: usize = 256;

// Two forward DIF levels (spans nn/2 then nn/4) fused into one load/store pass.
// Equivalent to two radix-2 passes; halves the plane traffic. Requires nn >= 32.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn ntt_plane_radix4(plane: &mut [u64], nn: usize, seg_base: usize, prime: usize, q: u64, q4: u64, powomega: &[u64]) {
    let n = plane.len();
    let ha = nn / 2;
    let qa = nn / 4;
    let count_a = ha - 1;
    let count_b = qa - 1;
    let a_om = seg_base;
    let a_qt = seg_base + 3 * count_a;
    let b_om = seg_base + 6 * count_a;
    let b_qt = b_om + 3 * count_b;
    let q4v = _mm512_set1_epi64(q4 as i64);
    let ptr = plane.as_mut_ptr();

    unsafe {
        let mut block = 0usize;
        while block < n {
            // i = 0: level-A first butterfly and both level-B butterflies are
            // identity; level-A second butterfly uses omega index qa - 1.
            {
                let x0 = plane[block];
                let x1 = plane[block + qa];
                let x2 = plane[block + ha];
                let x3 = plane[block + ha + qa];
                let lo0 = cond_sub_2q(x0 + x2, q4);
                let hi0 = cond_sub_2q(x0 + q4 - x2, q4);
                let lo1 = cond_sub_2q(x1 + x3, q4);
                let hi1 = harvey_modmul(
                    x1 + q4 - x3,
                    powomega[a_om + prime * count_a + (qa - 1)],
                    powomega[a_qt + prime * count_a + (qa - 1)],
                    q,
                );
                plane[block] = cond_sub_2q(lo0 + lo1, q4);
                plane[block + qa] = cond_sub_2q(lo0 + q4 - lo1, q4);
                plane[block + ha] = cond_sub_2q(hi0 + hi1, q4);
                plane[block + ha + qa] = cond_sub_2q(hi0 + q4 - hi1, q4);
            }

            let mut i = 1usize;
            while i + 8 <= qa {
                let p0 = block + i;
                let p1 = block + qa + i;
                let p2 = block + ha + i;
                let p3 = block + ha + qa + i;
                let x0 = _mm512_loadu_si512(ptr.add(p0) as *const __m512i);
                let x1 = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                let x2 = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                let x3 = _mm512_loadu_si512(ptr.add(p3) as *const __m512i);

                let lo0 = cond_sub_2q_si512(_mm512_add_epi64(x0, x2), q4v);
                let hi0 = harvey_modmul_plane8(
                    _mm512_sub_epi64(_mm512_add_epi64(x0, q4v), x2),
                    load_plane_twiddles8(powomega, a_om, count_a, i - 1, prime),
                    load_plane_twiddles8(powomega, a_qt, count_a, i - 1, prime),
                    q,
                );
                let lo1 = cond_sub_2q_si512(_mm512_add_epi64(x1, x3), q4v);
                let hi1 = harvey_modmul_plane8(
                    _mm512_sub_epi64(_mm512_add_epi64(x1, q4v), x3),
                    load_plane_twiddles8(powomega, a_om, count_a, i + qa - 1, prime),
                    load_plane_twiddles8(powomega, a_qt, count_a, i + qa - 1, prime),
                    q,
                );

                let b_omega = load_plane_twiddles8(powomega, b_om, count_b, i - 1, prime);
                let b_quot = load_plane_twiddles8(powomega, b_qt, count_b, i - 1, prime);
                let out0 = cond_sub_2q_si512(_mm512_add_epi64(lo0, lo1), q4v);
                let out1 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(lo0, q4v), lo1), b_omega, b_quot, q);
                let out2 = cond_sub_2q_si512(_mm512_add_epi64(hi0, hi1), q4v);
                let out3 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(hi0, q4v), hi1), b_omega, b_quot, q);

                _mm512_storeu_si512(ptr.add(p0) as *mut __m512i, out0);
                _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, out1);
                _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out2);
                _mm512_storeu_si512(ptr.add(p3) as *mut __m512i, out3);
                i += 8;
            }
            if i < qa {
                let r = qa - i;
                let mask = ((1u32 << r) - 1) as u8;
                let p0 = block + i;
                let p1 = block + qa + i;
                let p2 = block + ha + i;
                let p3 = block + ha + qa + i;
                let x0 = _mm512_maskz_loadu_epi64(mask, ptr.add(p0) as *const i64);
                let x1 = _mm512_maskz_loadu_epi64(mask, ptr.add(p1) as *const i64);
                let x2 = _mm512_maskz_loadu_epi64(mask, ptr.add(p2) as *const i64);
                let x3 = _mm512_maskz_loadu_epi64(mask, ptr.add(p3) as *const i64);

                let lo0 = cond_sub_2q_si512(_mm512_add_epi64(x0, x2), q4v);
                let hi0 = harvey_modmul_plane8(
                    _mm512_sub_epi64(_mm512_add_epi64(x0, q4v), x2),
                    maskz_load_plane_twiddles8(mask, powomega, a_om, count_a, i - 1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, a_qt, count_a, i - 1, prime),
                    q,
                );
                let lo1 = cond_sub_2q_si512(_mm512_add_epi64(x1, x3), q4v);
                let hi1 = harvey_modmul_plane8(
                    _mm512_sub_epi64(_mm512_add_epi64(x1, q4v), x3),
                    maskz_load_plane_twiddles8(mask, powomega, a_om, count_a, i + qa - 1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, a_qt, count_a, i + qa - 1, prime),
                    q,
                );

                let b_omega = maskz_load_plane_twiddles8(mask, powomega, b_om, count_b, i - 1, prime);
                let b_quot = maskz_load_plane_twiddles8(mask, powomega, b_qt, count_b, i - 1, prime);
                let out0 = cond_sub_2q_si512(_mm512_add_epi64(lo0, lo1), q4v);
                let out1 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(lo0, q4v), lo1), b_omega, b_quot, q);
                let out2 = cond_sub_2q_si512(_mm512_add_epi64(hi0, hi1), q4v);
                let out3 = harvey_modmul_plane8(_mm512_sub_epi64(_mm512_add_epi64(hi0, q4v), hi1), b_omega, b_quot, q);

                _mm512_mask_storeu_epi64(ptr.add(p0) as *mut i64, mask, out0);
                _mm512_mask_storeu_epi64(ptr.add(p1) as *mut i64, mask, out1);
                _mm512_mask_storeu_epi64(ptr.add(p2) as *mut i64, mask, out2);
                _mm512_mask_storeu_epi64(ptr.add(p3) as *mut i64, mask, out3);
            }

            block += nn;
        }
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn ntt_plane_avx512<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTable<P>, plane: &mut [u64], prime: usize) {
    let n = table.n;
    if n <= 1 {
        return;
    }
    debug_assert!(plane.len() >= n);

    unsafe {
        let q = P::Q[prime];
        let q2 = table.q2[prime];
        let q4 = table.q4[prime];
        let q2v = _mm512_set1_epi64(q2 as i64);
        let q4v = _mm512_set1_epi64(q4 as i64);
        let ptr = plane.as_mut_ptr();
        let powomega = table.powomega_plane.as_slice();
        let mut seg_base = 0usize;
        let mut nn = n;

        if n > 8 {
            ntt_plane_first_fused(plane, prime, q, q4, q4v, powomega);
            let halfn = n / 2;
            seg_base = 6 * n + 6 * (halfn - 1);
            nn = halfn;
        } else {
            let omega_base = seg_base;
            let quot_base = seg_base + 3 * n;
            let mut i = 0usize;
            while i + 8 <= n {
                let a = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
                let omega = load_plane_twiddles8(powomega, omega_base, n, i, prime);
                let omega_quot = load_plane_twiddles8(powomega, quot_base, n, i, prime);
                _mm512_storeu_si512(ptr.add(i) as *mut __m512i, harvey_modmul_plane8(a, omega, omega_quot, q));
                i += 8;
            }
            while i < n {
                plane[i] = harvey_modmul(
                    plane[i],
                    powomega[omega_base + prime * n + i],
                    powomega[quot_base + prime * n + i],
                    q,
                );
                i += 1;
            }
            seg_base += 6 * n;
        }

        while nn >= 32 {
            ntt_plane_radix4(plane, nn, seg_base, prime, q, q4, powomega);
            seg_base += 6 * (nn / 2 - 1) + 6 * (nn / 4 - 1);
            nn /= 4;
        }

        // Odd leftover upper level (nn == 16), then the radix-8 tail.
        while nn > 8 {
            let halfnn = nn / 2;

            let count = halfnn - 1;
            let omega_base = seg_base;
            let quot_base = seg_base + 3 * count;

            let mut block_start = 0usize;
            while block_start < n {
                {
                    let p1 = block_start;
                    let p2 = block_start + halfnn;
                    let a = plane[p1];
                    let b = plane[p2];
                    plane[p1] = cond_sub_2q(a + b, q4);
                    plane[p2] = cond_sub_2q(a + q4 - b, q4);
                }

                let mut i = 1usize;
                while i + 32 <= halfnn {
                    let p1 = block_start + i;
                    let p2 = block_start + halfnn + i;

                    let av0 = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                    let bv0 = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                    let av1 = _mm512_loadu_si512(ptr.add(p1 + 8) as *const __m512i);
                    let bv1 = _mm512_loadu_si512(ptr.add(p2 + 8) as *const __m512i);
                    let av2 = _mm512_loadu_si512(ptr.add(p1 + 16) as *const __m512i);
                    let bv2 = _mm512_loadu_si512(ptr.add(p2 + 16) as *const __m512i);
                    let av3 = _mm512_loadu_si512(ptr.add(p1 + 24) as *const __m512i);
                    let bv3 = _mm512_loadu_si512(ptr.add(p2 + 24) as *const __m512i);

                    let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bv0), q4v);
                    let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bv1), q4v);
                    let sum2 = cond_sub_2q_si512(_mm512_add_epi64(av2, bv2), q4v);
                    let sum3 = cond_sub_2q_si512(_mm512_add_epi64(av3, bv3), q4v);

                    let diff0 = _mm512_sub_epi64(_mm512_add_epi64(av0, q4v), bv0);
                    let diff1 = _mm512_sub_epi64(_mm512_add_epi64(av1, q4v), bv1);
                    let diff2 = _mm512_sub_epi64(_mm512_add_epi64(av2, q4v), bv2);
                    let diff3 = _mm512_sub_epi64(_mm512_add_epi64(av3, q4v), bv3);

                    let omega0 = load_plane_twiddles8(powomega, omega_base, count, i - 1, prime);
                    let omega1 = load_plane_twiddles8(powomega, omega_base, count, i + 7, prime);
                    let omega2 = load_plane_twiddles8(powomega, omega_base, count, i + 15, prime);
                    let omega3 = load_plane_twiddles8(powomega, omega_base, count, i + 23, prime);
                    let omega_quot0 = load_plane_twiddles8(powomega, quot_base, count, i - 1, prime);
                    let omega_quot1 = load_plane_twiddles8(powomega, quot_base, count, i + 7, prime);
                    let omega_quot2 = load_plane_twiddles8(powomega, quot_base, count, i + 15, prime);
                    let omega_quot3 = load_plane_twiddles8(powomega, quot_base, count, i + 23, prime);

                    let out0 = harvey_modmul_plane8(diff0, omega0, omega_quot0, q);
                    let out1 = harvey_modmul_plane8(diff1, omega1, omega_quot1, q);
                    let out2 = harvey_modmul_plane8(diff2, omega2, omega_quot2, q);
                    let out3 = harvey_modmul_plane8(diff3, omega3, omega_quot3, q);

                    _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum0);
                    _mm512_storeu_si512(ptr.add(p1 + 8) as *mut __m512i, sum1);
                    _mm512_storeu_si512(ptr.add(p1 + 16) as *mut __m512i, sum2);
                    _mm512_storeu_si512(ptr.add(p1 + 24) as *mut __m512i, sum3);
                    _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out0);
                    _mm512_storeu_si512(ptr.add(p2 + 8) as *mut __m512i, out1);
                    _mm512_storeu_si512(ptr.add(p2 + 16) as *mut __m512i, out2);
                    _mm512_storeu_si512(ptr.add(p2 + 24) as *mut __m512i, out3);

                    i += 32;
                }
                while i + 8 <= halfnn {
                    let p1 = block_start + i;
                    let p2 = block_start + halfnn + i;
                    let av = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                    let bv = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                    let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bv), q4v);
                    let diff = _mm512_sub_epi64(_mm512_add_epi64(av, q4v), bv);
                    let omega = load_plane_twiddles8(powomega, omega_base, count, i - 1, prime);
                    let omega_quot = load_plane_twiddles8(powomega, quot_base, count, i - 1, prime);
                    _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum);
                    _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, harvey_modmul_plane8(diff, omega, omega_quot, q));
                    i += 8;
                }
                if i < halfnn {
                    let r = halfnn - i;
                    let mask = ((1u32 << r) - 1) as u8;
                    let p1 = block_start + i;
                    let p2 = block_start + halfnn + i;
                    let av = _mm512_maskz_loadu_epi64(mask, ptr.add(p1) as *const i64);
                    let bv = _mm512_maskz_loadu_epi64(mask, ptr.add(p2) as *const i64);
                    let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bv), q4v);
                    let diff = _mm512_sub_epi64(_mm512_add_epi64(av, q4v), bv);
                    let omega = maskz_load_plane_twiddles8(mask, powomega, omega_base, count, i - 1, prime);
                    let omega_quot = maskz_load_plane_twiddles8(mask, powomega, quot_base, count, i - 1, prime);
                    let out = harvey_modmul_plane8(diff, omega, omega_quot, q);
                    _mm512_mask_storeu_epi64(ptr.add(p1) as *mut i64, mask, sum);
                    _mm512_mask_storeu_epi64(ptr.add(p2) as *mut i64, mask, out);
                }

                block_start += nn;
            }

            seg_base += 6 * count;
            nn /= 2;
        }

        if n >= 8 {
            ntt_plane_radix8_last3(plane, seg_base, prime, q, q4, powomega);
        } else {
            while nn >= 2 {
                let halfnn = nn / 2;

                if halfnn > 1 {
                    let count = halfnn - 1;
                    let omega_base = seg_base;
                    let quot_base = seg_base + 3 * count;

                    let mut block_start = 0usize;
                    while block_start < n {
                        {
                            let p1 = block_start;
                            let p2 = block_start + halfnn;
                            let a = plane[p1];
                            let b = plane[p2];
                            plane[p1] = cond_sub_2q(a + b, q4);
                            plane[p2] = cond_sub_2q(a + q4 - b, q4);
                        }

                        let mut i = 1usize;
                        while i < halfnn {
                            let p1 = block_start + i;
                            let p2 = block_start + halfnn + i;
                            let tw_idx = i - 1;
                            let a = plane[p1];
                            let b = plane[p2];
                            plane[p1] = cond_sub_2q(a + b, q4);
                            plane[p2] = harvey_modmul(
                                a + q4 - b,
                                powomega[omega_base + prime * count + tw_idx],
                                powomega[quot_base + prime * count + tw_idx],
                                q,
                            );
                            i += 1;
                        }

                        block_start += nn;
                    }

                    seg_base += 6 * count;
                } else {
                    let mut block_start = 0usize;
                    while block_start < n {
                        let a = plane[block_start];
                        let b = plane[block_start + 1];
                        plane[block_start] = cond_sub_2q(a + b, q4);
                        plane[block_start + 1] = cond_sub_2q(a + q4 - b, q4);
                        block_start += 2;
                    }
                }

                nn /= 2;
            }
        }

        // For n >= 8 the radix-8 tail already folds the [0,4q) -> [0,2q)
        // normalization into its output store; only the small-n path needs a
        // separate pass.
        if n < 8 {
            let mut i = 0usize;
            while i + 8 <= n {
                let x = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
                _mm512_storeu_si512(ptr.add(i) as *mut __m512i, cond_sub_2q_si512(x, q2v));
                i += 8;
            }
            while i < n {
                plane[i] = cond_sub_2q(plane[i], q2);
                i += 1;
            }
        }
    }
}

// Two inverse DIT levels (spans nn/2 then nn) fused into one load/store pass.
// Equivalent to two radix-2 passes; halves the plane traffic. Requires nn >= 16.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_plane_radix4(plane: &mut [u64], nn: usize, seg_base: usize, prime: usize, q: u64, q4: u64, powomega: &[u64]) {
    let n = plane.len();
    let sa = nn / 2;
    let sb = nn;
    let count_a = sa - 1;
    let count_b = sb - 1;
    let a_om = seg_base;
    let a_qt = seg_base + 3 * count_a;
    let b_om = seg_base + 6 * count_a;
    let b_qt = b_om + 3 * count_b;
    let q4v = _mm512_set1_epi64(q4 as i64);
    let ptr = plane.as_mut_ptr();

    unsafe {
        let mut block = 0usize;
        while block < n {
            // i = 0: level-A and the (p0,p2) level-B butterfly are identity;
            // the (p1,p3) level-B butterfly uses omega index sa - 1.
            {
                let x0 = plane[block];
                let x1 = plane[block + sa];
                let x2 = plane[block + sb];
                let x3 = plane[block + sb + sa];
                let u0 = cond_sub_2q(x0 + x1, q4);
                let u1 = cond_sub_2q(x0 + q4 - x1, q4);
                let u2 = cond_sub_2q(x2 + x3, q4);
                let u3 = cond_sub_2q(x2 + q4 - x3, q4);
                let bb = harvey_modmul(
                    u3,
                    powomega[b_om + prime * count_b + (sa - 1)],
                    powomega[b_qt + prime * count_b + (sa - 1)],
                    q,
                );
                plane[block] = cond_sub_2q(u0 + u2, q4);
                plane[block + sb] = cond_sub_2q(u0 + q4 - u2, q4);
                plane[block + sa] = cond_sub_2q(u1 + bb, q4);
                plane[block + sb + sa] = cond_sub_2q(u1 + q4 - bb, q4);
            }

            let mut i = 1usize;
            while i + 8 <= sa {
                let p0 = block + i;
                let p1 = block + sa + i;
                let p2 = block + sb + i;
                let p3 = block + sb + sa + i;
                let x0 = _mm512_loadu_si512(ptr.add(p0) as *const __m512i);
                let x1 = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                let x2 = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                let x3 = _mm512_loadu_si512(ptr.add(p3) as *const __m512i);

                let a_omega = load_plane_twiddles8(powomega, a_om, count_a, i - 1, prime);
                let a_quot = load_plane_twiddles8(powomega, a_qt, count_a, i - 1, prime);
                let ba0 = harvey_modmul_plane8(x1, a_omega, a_quot, q);
                let u0 = cond_sub_2q_si512(_mm512_add_epi64(x0, ba0), q4v);
                let u1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(x0, q4v), ba0), q4v);
                let ba1 = harvey_modmul_plane8(x3, a_omega, a_quot, q);
                let u2 = cond_sub_2q_si512(_mm512_add_epi64(x2, ba1), q4v);
                let u3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(x2, q4v), ba1), q4v);

                let bb0 = harvey_modmul_plane8(
                    u2,
                    load_plane_twiddles8(powomega, b_om, count_b, i - 1, prime),
                    load_plane_twiddles8(powomega, b_qt, count_b, i - 1, prime),
                    q,
                );
                let out0 = cond_sub_2q_si512(_mm512_add_epi64(u0, bb0), q4v);
                let out2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u0, q4v), bb0), q4v);
                let bb1 = harvey_modmul_plane8(
                    u3,
                    load_plane_twiddles8(powomega, b_om, count_b, i + sa - 1, prime),
                    load_plane_twiddles8(powomega, b_qt, count_b, i + sa - 1, prime),
                    q,
                );
                let out1 = cond_sub_2q_si512(_mm512_add_epi64(u1, bb1), q4v);
                let out3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u1, q4v), bb1), q4v);

                _mm512_storeu_si512(ptr.add(p0) as *mut __m512i, out0);
                _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, out1);
                _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out2);
                _mm512_storeu_si512(ptr.add(p3) as *mut __m512i, out3);
                i += 8;
            }
            if i < sa {
                let r = sa - i;
                let mask = ((1u32 << r) - 1) as u8;
                let p0 = block + i;
                let p1 = block + sa + i;
                let p2 = block + sb + i;
                let p3 = block + sb + sa + i;
                let x0 = _mm512_maskz_loadu_epi64(mask, ptr.add(p0) as *const i64);
                let x1 = _mm512_maskz_loadu_epi64(mask, ptr.add(p1) as *const i64);
                let x2 = _mm512_maskz_loadu_epi64(mask, ptr.add(p2) as *const i64);
                let x3 = _mm512_maskz_loadu_epi64(mask, ptr.add(p3) as *const i64);

                let a_omega = maskz_load_plane_twiddles8(mask, powomega, a_om, count_a, i - 1, prime);
                let a_quot = maskz_load_plane_twiddles8(mask, powomega, a_qt, count_a, i - 1, prime);
                let ba0 = harvey_modmul_plane8(x1, a_omega, a_quot, q);
                let u0 = cond_sub_2q_si512(_mm512_add_epi64(x0, ba0), q4v);
                let u1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(x0, q4v), ba0), q4v);
                let ba1 = harvey_modmul_plane8(x3, a_omega, a_quot, q);
                let u2 = cond_sub_2q_si512(_mm512_add_epi64(x2, ba1), q4v);
                let u3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(x2, q4v), ba1), q4v);

                let bb0 = harvey_modmul_plane8(
                    u2,
                    maskz_load_plane_twiddles8(mask, powomega, b_om, count_b, i - 1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, b_qt, count_b, i - 1, prime),
                    q,
                );
                let out0 = cond_sub_2q_si512(_mm512_add_epi64(u0, bb0), q4v);
                let out2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u0, q4v), bb0), q4v);
                let bb1 = harvey_modmul_plane8(
                    u3,
                    maskz_load_plane_twiddles8(mask, powomega, b_om, count_b, i + sa - 1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, b_qt, count_b, i + sa - 1, prime),
                    q,
                );
                let out1 = cond_sub_2q_si512(_mm512_add_epi64(u1, bb1), q4v);
                let out3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(u1, q4v), bb1), q4v);

                _mm512_mask_storeu_epi64(ptr.add(p0) as *mut i64, mask, out0);
                _mm512_mask_storeu_epi64(ptr.add(p1) as *mut i64, mask, out1);
                _mm512_mask_storeu_epi64(ptr.add(p2) as *mut i64, mask, out2);
                _mm512_mask_storeu_epi64(ptr.add(p3) as *mut i64, mask, out3);
            }

            block += 2 * sb;
        }
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_plane_avx512<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTableInv<P>, plane: &mut [u64], prime: usize) {
    let n = table.n;
    if n <= 1 {
        return;
    }
    debug_assert!(plane.len() >= n);

    unsafe {
        let q = P::Q[prime];
        let q4 = table.q4[prime];
        let q4v = _mm512_set1_epi64(q4 as i64);
        let ptr = plane.as_mut_ptr();
        let powomega = table.powomega_plane.as_slice();
        let mut seg_base = 0usize;

        let mut nn = 2usize;
        if n >= 8 {
            intt_plane_radix8_first3(plane, seg_base, prime, q, q4, powomega);
            seg_base += 24;
            nn = 16;
        }

        // Fuse the level-0 untwist into the last butterfly level (nn == n).
        let fuse_untwist = n >= 16;

        while nn <= n / 4 {
            intt_plane_radix4(plane, nn, seg_base, prime, q, q4, powomega);
            seg_base += 6 * (nn / 2 - 1) + 6 * (nn - 1);
            nn *= 4;
        }

        while nn <= n {
            if fuse_untwist && nn == n {
                break;
            }
            let halfnn = nn / 2;

            if halfnn > 1 {
                let count = halfnn - 1;
                let omega_base = seg_base;
                let quot_base = seg_base + 3 * count;

                let mut block_start = 0usize;
                while block_start < n {
                    {
                        let p1 = block_start;
                        let p2 = block_start + halfnn;
                        let a = plane[p1];
                        let b = plane[p2];
                        plane[p1] = cond_sub_2q(a + b, q4);
                        plane[p2] = cond_sub_2q(a + q4 - b, q4);
                    }

                    let mut i = 1usize;
                    while i + 32 <= halfnn {
                        let p1 = block_start + i;
                        let p2 = block_start + halfnn + i;

                        let av0 = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                        let bv0 = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                        let av1 = _mm512_loadu_si512(ptr.add(p1 + 8) as *const __m512i);
                        let bv1 = _mm512_loadu_si512(ptr.add(p2 + 8) as *const __m512i);
                        let av2 = _mm512_loadu_si512(ptr.add(p1 + 16) as *const __m512i);
                        let bv2 = _mm512_loadu_si512(ptr.add(p2 + 16) as *const __m512i);
                        let av3 = _mm512_loadu_si512(ptr.add(p1 + 24) as *const __m512i);
                        let bv3 = _mm512_loadu_si512(ptr.add(p2 + 24) as *const __m512i);

                        let omega0 = load_plane_twiddles8(powomega, omega_base, count, i - 1, prime);
                        let omega1 = load_plane_twiddles8(powomega, omega_base, count, i + 7, prime);
                        let omega2 = load_plane_twiddles8(powomega, omega_base, count, i + 15, prime);
                        let omega3 = load_plane_twiddles8(powomega, omega_base, count, i + 23, prime);
                        let omega_quot0 = load_plane_twiddles8(powomega, quot_base, count, i - 1, prime);
                        let omega_quot1 = load_plane_twiddles8(powomega, quot_base, count, i + 7, prime);
                        let omega_quot2 = load_plane_twiddles8(powomega, quot_base, count, i + 15, prime);
                        let omega_quot3 = load_plane_twiddles8(powomega, quot_base, count, i + 23, prime);

                        let bo0 = harvey_modmul_plane8(bv0, omega0, omega_quot0, q);
                        let bo1 = harvey_modmul_plane8(bv1, omega1, omega_quot1, q);
                        let bo2 = harvey_modmul_plane8(bv2, omega2, omega_quot2, q);
                        let bo3 = harvey_modmul_plane8(bv3, omega3, omega_quot3, q);

                        let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bo0), q4v);
                        let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bo1), q4v);
                        let sum2 = cond_sub_2q_si512(_mm512_add_epi64(av2, bo2), q4v);
                        let sum3 = cond_sub_2q_si512(_mm512_add_epi64(av3, bo3), q4v);

                        let diff0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av0, q4v), bo0), q4v);
                        let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av1, q4v), bo1), q4v);
                        let diff2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av2, q4v), bo2), q4v);
                        let diff3 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av3, q4v), bo3), q4v);

                        _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum0);
                        _mm512_storeu_si512(ptr.add(p1 + 8) as *mut __m512i, sum1);
                        _mm512_storeu_si512(ptr.add(p1 + 16) as *mut __m512i, sum2);
                        _mm512_storeu_si512(ptr.add(p1 + 24) as *mut __m512i, sum3);
                        _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, diff0);
                        _mm512_storeu_si512(ptr.add(p2 + 8) as *mut __m512i, diff1);
                        _mm512_storeu_si512(ptr.add(p2 + 16) as *mut __m512i, diff2);
                        _mm512_storeu_si512(ptr.add(p2 + 24) as *mut __m512i, diff3);

                        i += 32;
                    }
                    while i + 8 <= halfnn {
                        let p1 = block_start + i;
                        let p2 = block_start + halfnn + i;
                        let av = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                        let bv = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                        let omega = load_plane_twiddles8(powomega, omega_base, count, i - 1, prime);
                        let omega_quot = load_plane_twiddles8(powomega, quot_base, count, i - 1, prime);
                        let bo = harvey_modmul_plane8(bv, omega, omega_quot, q);
                        let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4v);
                        let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4v), bo), q4v);
                        _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, sum);
                        _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, diff);
                        i += 8;
                    }
                    if i < halfnn {
                        let r = halfnn - i;
                        let mask = ((1u32 << r) - 1) as u8;
                        let p1 = block_start + i;
                        let p2 = block_start + halfnn + i;
                        let av = _mm512_maskz_loadu_epi64(mask, ptr.add(p1) as *const i64);
                        let bv = _mm512_maskz_loadu_epi64(mask, ptr.add(p2) as *const i64);
                        let omega = maskz_load_plane_twiddles8(mask, powomega, omega_base, count, i - 1, prime);
                        let omega_quot = maskz_load_plane_twiddles8(mask, powomega, quot_base, count, i - 1, prime);
                        let bo = harvey_modmul_plane8(bv, omega, omega_quot, q);
                        let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4v);
                        let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4v), bo), q4v);
                        _mm512_mask_storeu_epi64(ptr.add(p1) as *mut i64, mask, sum);
                        _mm512_mask_storeu_epi64(ptr.add(p2) as *mut i64, mask, diff);
                    }

                    block_start += nn;
                }

                seg_base += 6 * count;
            } else {
                let mut block_start = 0usize;
                while block_start < n {
                    let a = plane[block_start];
                    let b = plane[block_start + 1];
                    plane[block_start] = cond_sub_2q(a + b, q4);
                    plane[block_start + 1] = cond_sub_2q(a + q4 - b, q4);
                    block_start += 2;
                }
            }

            nn *= 2;
        }

        if fuse_untwist {
            // Last level (nn = n, single block) fused with the level-0 untwist:
            // the butterfly outputs are multiplied by their untwist twiddle on
            // the way to memory, removing a separate full pass over the plane.
            let halfnn = n / 2;
            let count = halfnn - 1;
            let bf_omega = seg_base;
            let bf_quot = seg_base + 3 * count;
            let ut_omega = seg_base + 6 * count;
            let ut_quot = ut_omega + 3 * n;

            // i = 0: butterfly without twiddle, then untwist both outputs.
            let a = plane[0];
            let b = plane[halfnn];
            plane[0] = harvey_modmul(
                cond_sub_2q(a + b, q4),
                powomega[ut_omega + prime * n],
                powomega[ut_quot + prime * n],
                q,
            );
            plane[halfnn] = harvey_modmul(
                cond_sub_2q(a + q4 - b, q4),
                powomega[ut_omega + prime * n + halfnn],
                powomega[ut_quot + prime * n + halfnn],
                q,
            );

            let mut i = 1usize;
            while i + 8 <= halfnn {
                let p1 = i;
                let p2 = halfnn + i;
                let av = _mm512_loadu_si512(ptr.add(p1) as *const __m512i);
                let bv = _mm512_loadu_si512(ptr.add(p2) as *const __m512i);
                let bo = harvey_modmul_plane8(
                    bv,
                    load_plane_twiddles8(powomega, bf_omega, count, i - 1, prime),
                    load_plane_twiddles8(powomega, bf_quot, count, i - 1, prime),
                    q,
                );
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4v);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4v), bo), q4v);
                let out_lo = harvey_modmul_plane8(
                    sum,
                    load_plane_twiddles8(powomega, ut_omega, n, p1, prime),
                    load_plane_twiddles8(powomega, ut_quot, n, p1, prime),
                    q,
                );
                let out_hi = harvey_modmul_plane8(
                    diff,
                    load_plane_twiddles8(powomega, ut_omega, n, p2, prime),
                    load_plane_twiddles8(powomega, ut_quot, n, p2, prime),
                    q,
                );
                _mm512_storeu_si512(ptr.add(p1) as *mut __m512i, out_lo);
                _mm512_storeu_si512(ptr.add(p2) as *mut __m512i, out_hi);
                i += 8;
            }
            if i < halfnn {
                let r = halfnn - i;
                let mask = ((1u32 << r) - 1) as u8;
                let p1 = i;
                let p2 = halfnn + i;
                let av = _mm512_maskz_loadu_epi64(mask, ptr.add(p1) as *const i64);
                let bv = _mm512_maskz_loadu_epi64(mask, ptr.add(p2) as *const i64);
                let bo = harvey_modmul_plane8(
                    bv,
                    maskz_load_plane_twiddles8(mask, powomega, bf_omega, count, i - 1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, bf_quot, count, i - 1, prime),
                    q,
                );
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4v);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4v), bo), q4v);
                let out_lo = harvey_modmul_plane8(
                    sum,
                    maskz_load_plane_twiddles8(mask, powomega, ut_omega, n, p1, prime),
                    maskz_load_plane_twiddles8(mask, powomega, ut_quot, n, p1, prime),
                    q,
                );
                let out_hi = harvey_modmul_plane8(
                    diff,
                    maskz_load_plane_twiddles8(mask, powomega, ut_omega, n, p2, prime),
                    maskz_load_plane_twiddles8(mask, powomega, ut_quot, n, p2, prime),
                    q,
                );
                _mm512_mask_storeu_epi64(ptr.add(p1) as *mut i64, mask, out_lo);
                _mm512_mask_storeu_epi64(ptr.add(p2) as *mut i64, mask, out_hi);
            }
        } else {
            let omega_base = seg_base;
            let quot_base = seg_base + 3 * n;
            let mut i = 0usize;
            while i + 8 <= n {
                let a = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
                let omega = load_plane_twiddles8(powomega, omega_base, n, i, prime);
                let omega_quot = load_plane_twiddles8(powomega, quot_base, n, i, prime);
                _mm512_storeu_si512(ptr.add(i) as *mut __m512i, harvey_modmul_plane8(a, omega, omega_quot, q));
                i += 8;
            }
            while i < n {
                plane[i] = harvey_modmul(
                    plane[i],
                    powomega[omega_base + prime * n + i],
                    powomega[quot_base + prime * n + i],
                    q,
                );
                i += 1;
            }
        }
    }
}

/// Forward NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// Butterfly values live in `[0, 4q)`; a final pass renormalises to `[0, 2q)`.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn ntt_avx512<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTable<P>, data: &mut [u64]) {
    let n = table.n;
    debug_assert!(data.len() >= 3 * n);
    unsafe {
        for prime in 0..3 {
            ntt_plane_avx512::<P>(table, &mut data[prime * n..(prime + 1) * n], prime);
        }
    }
}

/// Forward NTT without the final `[0, 4q) -> [0, 2q)` normalisation pass.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn ntt_avx512_no_final<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTable<P>, data: &mut [u64]) {
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

        let mut seg_avx = 0usize;
        let block = NTT_BLOCK.min(n);
        let mut nn = n;

        if n > block {
            // Level 0 (a[i] *= ω^i) fused with the first butterfly level.
            let halfn = n / 2;
            let count = halfn - 1;
            ntt_iter_first_fused_ifma(
                begin,
                halfn,
                po_base,
                po_base.add(n),
                po_base.add(2 * n),
                po_base.add(2 * n + count),
                q,
                q4,
            );
            seg_avx = 2 * n + 2 * count;
            nn = halfn;
        } else {
            // Level 0: a[i] *= ω^i.
            ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
            seg_avx += 2 * n;
        }

        // Upper butterfly levels (breadth-first) while nn > NTT_BLOCK.
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
                let seg = inner_segs[i];
                if m == 8 {
                    // Levels nn = 8, 4, 2 fused in one register pass.
                    let seg4 = inner_segs[i + 1];
                    ntt_radix8_last3_ifma(
                        blk_begin,
                        blk_end,
                        q,
                        q4,
                        po_base.add(seg),
                        po_base.add(seg + 3),
                        po_base.add(seg4),
                        po_base.add(seg4 + 1),
                    );
                    break;
                }
                let count = m / 2 - 1;
                ntt_iter_ifma(m, blk_begin, blk_end, q, q4, po_base.add(seg), po_base.add(seg + count));
            }
            blk_start += nn;
        }
    }
}

/// [`ntt_avx512`] variant that writes the final normalised x2-blocks to
/// strided rows: block `i` lands at `dst[i * row_stride + row_off]` (u64
/// units). Used by the convolution prepare to skip a separate scatter pass.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn ntt_avx512_to_rows<P: PrimeSetNtt126Ifma>(
    table: &Ntt126IfmaTable<P>,
    data: &mut [u64],
    dst: &mut [u64],
    row_stride: usize,
    row_off: usize,
) {
    unsafe {
        let n = table.n;
        ntt_avx512::<P>(table, data);
        for prime in 0..3 {
            let src = &data[prime * n..(prime + 1) * n];
            let plane_off = row_off + prime * row_stride * (n / 8);
            let mut i = 0usize;
            while i < n / 8 {
                let x = _mm512_loadu_si512(src.as_ptr().add(8 * i) as *const __m512i);
                let out = dst.as_mut_ptr().add(plane_off + i * row_stride) as *mut __m512i;
                _mm512_storeu_si512(out, x);
                i += 1;
            }
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
    debug_assert!(data.len() >= 3 * n);
    unsafe {
        for prime in 0..3 {
            intt_plane_avx512::<P>(table, &mut data[prime * n..(prime + 1) * n], prime);
        }
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

            let mut data_avx = vec![0u64; 3 * n];
            let mut data_ref = vec![0u64; 3 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
            b_ntt126_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx512::<Primes42>(&fwd, &mut data_avx) };
            ntt126_ifma_ref::<Primes42>(&fwd, &mut data_ref);

            for i in 0..3 * n {
                assert_eq!(
                    data_avx[i], data_ref[i],
                    "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i], data_ref[i]
                );
            }
        }
    }

    #[test]
    fn ntt_avx512_vs_ref_n4096_pseudorandom() {
        let n = 4096usize;
        let fwd = Ntt126IfmaTable::<Primes42>::new(n);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let coeffs: Vec<i64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 11) as i64 % 20001) - 10000
            })
            .collect();

        let mut data_avx = vec![0u64; 3 * n];
        let mut data_ref = vec![0u64; 3 * n];
        b_ntt126_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
        b_ntt126_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

        unsafe { ntt_avx512::<Primes42>(&fwd, &mut data_avx) };
        ntt126_ifma_ref::<Primes42>(&fwd, &mut data_ref);

        for i in 0..3 * n {
            assert_eq!(
                data_avx[i], data_ref[i],
                "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                data_avx[i], data_ref[i]
            );
        }
    }

    #[test]
    fn intt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt126IfmaTable::<Primes42>::new(n);
            let inv = Ntt126IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 3 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data, &coeffs);
            ntt126_ifma_ref::<Primes42>(&fwd, &mut data);

            let mut data_avx = data.clone();
            let mut data_ref = data.clone();

            unsafe { intt_avx512::<Primes42>(&inv, &mut data_avx) };
            intt126_ifma_ref::<Primes42>(&inv, &mut data_ref);

            for i in 0..3 * n {
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
            let mut data = vec![0u64; 3 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let orig = data.clone();

            unsafe {
                ntt_avx512::<Primes42>(&fwd, &mut data);
                intt_avx512::<Primes42>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..3 {
                    let o = orig[k * n + i] % Primes42::Q[k];
                    let g = data[k * n + i] % Primes42::Q[k];
                    assert_eq!(o, g, "n={n} i={i} k={k}: roundtrip mismatch");
                }
            }
        }
    }
}
