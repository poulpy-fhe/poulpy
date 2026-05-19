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

//! AVX-512F accelerated coefficient-domain Q120 arithmetic.
//!
//! Provides the kernels used by [`super::prim`]:
//!
//! | Function | Trait |
//! |---|---|
//! | [`b_from_znx64_avx512`] | `NttFromZnx64` |
//! | [`b_from_znx64_masked_avx512`] | `NttFromZnx64::ntt_from_znx64_masked` |
//! | [`c_from_b_avx512`] | `NttCFromB` |
//! | [`vec_mat1col_product_bbb_avx512`] | `NttMulBbb` |
//! | [`b_to_znx128_avx512`] | `NttToZnx128` |
//!
//! Inner loops pair-pack two q120b coefficients per `__m512i`; per-prime
//! constants are broadcast to both halves with `_mm512_broadcast_i64x4`. A
//! 256-bit tail handles the odd-coefficient case.
//!
//! All functions are gated on `#[target_feature(enable = "avx512f")]` and
//! marked `pub(crate) unsafe fn`; the caller (trait impls in `prim.rs`) must
//! have verified CPU support at module construction time.

use core::arch::x86_64::{
    __m256i, __m512i, _mm_add_epi64, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_unpackhi_epi64, _mm256_add_epi64,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi64, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi64,
    _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64, _mm512_add_epi32, _mm512_add_epi64,
    _mm512_and_si512, _mm512_broadcast_i64x4, _mm512_cmpgt_epi64_mask, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
    _mm512_mask_add_epi64, _mm512_mask_sub_epi64, _mm512_mul_epu32, _mm512_or_si512, _mm512_set_epi64, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
};

use poulpy_cpu_ref::reference::ntt120::{
    mat_vec::BbbMeta,
    primes::{PrimeSet, Primes30},
};

// ─────────────────────────────────────────────────────────────────────────────
// Primes30-specific compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

/// `Q[k]` as `u64`, one per prime, broadcast across SIMD lanes.
pub(crate) const Q_VEC: [u64; 4] = [
    Primes30::Q[0] as u64,
    Primes30::Q[1] as u64,
    Primes30::Q[2] as u64,
    Primes30::Q[3] as u64,
];

/// `oq[k] = Q[k] - (2^63 mod Q[k])`.
///
/// Used by `b_from_znx64_avx512`: for a negative input `x`, each prime lane
/// receives `(x as u64 & i64::MAX) + oq[k]`, which equals `x mod Q[k]` as u64.
pub(crate) const OQ: [u64; 4] = {
    let mut oq = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        let q = Q_VEC[k];
        oq[k] = q - (i64::MIN as u64 % q); // i64::MIN as u64 = 2^63
        k += 1;
    }
    oq
};

/// Barrett multiplier: `mu[k] = floor(2^61 / Q[k])`.
///
/// Used for Barrett reduction of values `x < 2^61` mod `Q[k]`.
/// Since `Q[k] > 2^29` for Primes30, `mu[k] < 2^32` (fits in u32 / lower 32 bits of u64).
pub(crate) const BARRETT_MU: [u64; 4] = {
    let mut mu = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        mu[k] = (1u64 << 61) / Q_VEC[k];
        k += 1;
    }
    mu
};

/// `pow32[k] = 2^32 mod Q[k]`.
///
/// Used in `c_from_b_avx512` and `b_to_znx128_avx512`:
/// - Combines `x_hi_r * pow32 + x_lo` to reduce a 63-bit q120b value.
/// - Computes `r_shift = r * pow32 mod Q[k]` (i.e., `r * 2^32 mod Q[k]`).
pub(crate) const POW32: [u64; 4] = {
    let mut p = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        p[k] = ((1u128 << 32) % Q_VEC[k] as u128) as u64;
        k += 1;
    }
    p
};

/// `CRT_CST[k]` as u64, for `b_to_znx128_avx512`.
pub(crate) const CRT_VEC: [u64; 4] = [
    Primes30::CRT_CST[0] as u64,
    Primes30::CRT_CST[1] as u64,
    Primes30::CRT_CST[2] as u64,
    Primes30::CRT_CST[3] as u64,
];

/// `pow32_crt[k] = (pow32[k] * CRT_CST[k]) mod Q[k]`.
///
/// Used by [`reduce_b_and_apply_crt`]: folds the high-word Barrett step and CRT multiply into
/// a single constant, so the contribution of `x_hi_r` (upper 32 bits of a q120b value, reduced
/// mod Q) directly maps to a CRT-weighted residue without an intermediate Barrett pass.
pub(crate) const POW32_CRT: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        r[k] = (POW32[k] * CRT_VEC[k]) % Q_VEC[k];
        k += 1;
    }
    r
};

/// `pow16_crt[k] = (2^16 mod Q[k]) * CRT_CST[k] mod Q[k]`.
///
/// Used by [`reduce_b_and_apply_crt`]: handles the middle 16 bits of `x_lo` when the full
/// `x_lo * CRT_CST` product would exceed `2^61` (the Barrett bound). Since `Q > 2^29 > 2^16`,
/// `2^16 mod Q[k] = 2^16` exactly, so this is just `(65536 * CRT_CST[k]) mod Q[k]`.
pub(crate) const POW16_CRT: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        // Q[k] > 2^29 > 2^16, so 2^16 mod Q[k] = 2^16 exactly.
        r[k] = ((1u64 << 16) * CRT_VEC[k]) % Q_VEC[k];
        k += 1;
    }
    r
};

// ─────────────────────────────────────────────────────────────────────────────
// CRT accumulation constants
// ─────────────────────────────────────────────────────────────────────────────

/// `qm[k] = total_q / Q[k]` as u128 (product of the three complementary primes).
const QM: [u128; 4] = {
    let q0 = Primes30::Q[0] as u128;
    let q1 = Primes30::Q[1] as u128;
    let q2 = Primes30::Q[2] as u128;
    let q3 = Primes30::Q[3] as u128;
    [q1 * q2 * q3, q0 * q2 * q3, q0 * q1 * q3, q0 * q1 * q2]
};

/// High 64-bit limb of `qm[k]`: `QM_HI[k] = qm[k] >> 64`.
///
/// `qm[k] < (2^30)^3 = 2^90`, so `QM_HI[k] < 2^26` — fits in 32 bits,
/// enabling `_mm256_mul_epu32(t, QM_HI_VEC)` without overflow.
pub(crate) const QM_HI: [u64; 4] = [
    (QM[0] >> 64) as u64,
    (QM[1] >> 64) as u64,
    (QM[2] >> 64) as u64,
    (QM[3] >> 64) as u64,
];

/// Middle 32-bit limb of `qm[k]`: `(qm[k] >> 32) & 0xFFFF_FFFF`.
pub(crate) const QM_MID: [u64; 4] = [
    ((QM[0] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[1] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[2] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[3] >> 32) & 0xFFFF_FFFF) as u64,
];

/// Low 32-bit limb of `qm[k]`: `qm[k] & 0xFFFF_FFFF`.
pub(crate) const QM_LO: [u64; 4] = [
    (QM[0] & 0xFFFF_FFFF) as u64,
    (QM[1] & 0xFFFF_FFFF) as u64,
    (QM[2] & 0xFFFF_FFFF) as u64,
    (QM[3] & 0xFFFF_FFFF) as u64,
];

/// `total_q = Q[0] * Q[1] * Q[2] * Q[3]` as u128.
pub(crate) const TOTAL_Q: u128 = {
    let q0 = Primes30::Q[0] as u128;
    let q1 = Primes30::Q[1] as u128;
    let q2 = Primes30::Q[2] as u128;
    let q3 = Primes30::Q[3] as u128;
    q0 * q1 * q2 * q3
};

/// `[0, total_q, 2·total_q, 3·total_q]` — lookup table for table-based modular reduction.
///
/// Replaces 3 conditional subtracts with 1 shift + 1 table load + 1 unconditional subtract
/// + at most 1 correction subtract (proved: `q_real - q_approx ≤ 1` for Primes30).
pub(crate) const TOTAL_Q_MULT: [u128; 4] = [0, TOTAL_Q, TOTAL_Q * 2, TOTAL_Q * 3];

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512F helpers (2 q120b coefficients per __m512i; per-prime constants
// broadcast to both 256-bit halves via `_mm512_broadcast_i64x4`).
// ─────────────────────────────────────────────────────────────────────────────

/// Broadcast a 4 × u64 per-prime constant to both halves of an `__m512i`.
#[inline(always)]
pub(crate) unsafe fn bcast_quad(p: *const u64) -> __m512i {
    unsafe { _mm512_broadcast_i64x4(_mm256_loadu_si256(p as *const __m256i)) }
}

/// Single conditional subtract (AVX-512F): `x = if x >=u q { x - q } else { x }`.
///
/// Mask form: `_mm512_cmpgt_epi64_mask(q, x)` returns bit `i` set when `q_i > x_i`
/// (signed). Valid when both `x < 2^63` and `q < 2^63` so signed cmp matches unsigned
/// order. Subtract `q` only where the mask is **clear** (i.e., `x >= q`).
#[inline(always)]
pub(crate) unsafe fn cond_sub_512(x: __m512i, q: __m512i) -> __m512i {
    unsafe {
        let lt_mask = _mm512_cmpgt_epi64_mask(q, x);
        _mm512_mask_sub_epi64(x, !lt_mask, x, q)
    }
}

/// Barrett reduction (AVX-512F): reduce `tmp < 2^61` to `[0, Q[k])` per prime,
/// across 8 lanes (= 2 coefficients).
#[inline(always)]
pub(crate) unsafe fn barrett_reduce_512(tmp: __m512i, q: __m512i, mu: __m512i) -> __m512i {
    unsafe {
        let mask32 = _mm512_set1_epi64(u32::MAX as i64);
        let tmp_hi = _mm512_srli_epi64::<32>(tmp);
        let tmp_lo = _mm512_and_si512(tmp, mask32);
        let q_hi = _mm512_srli_epi64::<29>(_mm512_mul_epu32(tmp_hi, mu));
        let q_lo = _mm512_srli_epi64::<61>(_mm512_mul_epu32(tmp_lo, mu));
        let q_approx = _mm512_add_epi64(q_hi, q_lo);
        let r = _mm512_sub_epi64(tmp, _mm512_mul_epu32(q_approx, q));
        let r = cond_sub_512(r, q);
        cond_sub_512(r, q)
    }
}

/// Reduce a pair of q120b values to canonical residues `[0, Q[k])` (8 lanes).
#[inline(always)]
pub(crate) unsafe fn reduce_b_to_canonical_512(x: __m512i, q: __m512i, mu: __m512i, pow32: __m512i) -> __m512i {
    unsafe {
        let mask32 = _mm512_set1_epi64(u32::MAX as i64);
        let x_hi = _mm512_srli_epi64::<32>(x);
        let x_lo = _mm512_and_si512(x, mask32);
        let x_hi_r = cond_sub_512(x_hi, q);
        let tmp = _mm512_add_epi64(_mm512_mul_epu32(x_hi_r, pow32), x_lo);
        barrett_reduce_512(tmp, q, mu)
    }
}

/// Fused q120b reduce + CRT multiply (AVX-512F): `t[k] = (x * CRT_CST[k]) mod Q[k]`
/// computed in a single Barrett pass across 8 lanes (= 2 coefficients).
#[inline(always)]
pub(crate) unsafe fn reduce_b_and_apply_crt_512(
    x: __m512i,
    q: __m512i,
    mu: __m512i,
    pow32_crt: __m512i,
    pow16_crt: __m512i,
    crt: __m512i,
) -> __m512i {
    unsafe {
        let mask32 = _mm512_set1_epi64(u32::MAX as i64);
        let mask16 = _mm512_set1_epi64(0xFFFF_i64);
        let x_hi = _mm512_srli_epi64::<32>(x);
        let x_hi_r = cond_sub_512(x_hi, q);
        let x_lo = _mm512_and_si512(x, mask32);
        let x_lo_hi = _mm512_srli_epi64::<16>(x_lo);
        let x_lo_lo = _mm512_and_si512(x_lo, mask16);
        let p1 = _mm512_mul_epu32(x_hi_r, pow32_crt);
        let p2 = _mm512_mul_epu32(x_lo_hi, pow16_crt);
        let p3 = _mm512_mul_epu32(x_lo_lo, crt);
        let tmp = _mm512_add_epi64(_mm512_add_epi64(p1, p2), p3);
        barrett_reduce_512(tmp, q, mu)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 256-bit helpers (used for the odd-coefficient tail)
// ─────────────────────────────────────────────────────────────────────────────

/// Single conditional subtract: `x = if x >= q { x - q } else { x }`.
///
/// Valid when both `x < 2^63` and `q < 2^63` (signed cmpgt gives correct unsigned order).
#[inline(always)]
pub(crate) unsafe fn cond_sub(x: __m256i, q: __m256i) -> __m256i {
    unsafe {
        // lt = all-ones in lanes where q > x (i.e., x < q — no subtract needed)
        let lt = _mm256_cmpgt_epi64(q, x);
        _mm256_sub_epi64(x, _mm256_andnot_si256(lt, q))
    }
}

/// Barrett reduction: reduce `tmp < 2^61` to `[0, Q[k])` for all four Primes30 lanes.
///
/// Uses precomputed `mu[k] = floor(2^61 / Q[k])` (stored in lower 32 bits of each u64 lane).
/// The quotient approximation may underestimate the true quotient by up to 2,
/// so two conditional subtracts bring the remainder into `[0, Q)`.
#[inline(always)]
pub(crate) unsafe fn barrett_reduce(tmp: __m256i, q: __m256i, mu: __m256i) -> __m256i {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // Split tmp at bit 32: tmp_hi < 2^29, tmp_lo < 2^32
        let tmp_hi = _mm256_srli_epi64::<32>(tmp);
        let tmp_lo = _mm256_and_si256(tmp, mask32);
        // q_approx_hi = floor(tmp_hi * mu / 2^29)
        //   tmp_hi * mu < 2^29 * 2^32 = 2^61, fits in u64
        let q_hi = _mm256_srli_epi64::<29>(_mm256_mul_epu32(tmp_hi, mu));
        // q_approx_lo = floor(tmp_lo * mu / 2^61)
        //   tmp_lo * mu < 2^32 * 2^32 = 2^64, may overflow — clamp contribution to 0..7
        let q_lo = _mm256_srli_epi64::<61>(_mm256_mul_epu32(tmp_lo, mu));
        let q_approx = _mm256_add_epi64(q_hi, q_lo);
        // r = tmp - q_approx * Q  (q_approx < 2^31, Q < 2^30, product < 2^61)
        let r = _mm256_sub_epi64(tmp, _mm256_mul_epu32(q_approx, q));
        // r < 3*Q after the approximation; two subtracts bring it into [0, Q)
        let r = cond_sub(r, q);
        cond_sub(r, q)
    }
}

/// Horizontal sum of 4 × u64 lanes in a `__m256i`.
///
/// Returns `lane[0] + lane[1] + lane[2] + lane[3]` as u64.
#[inline(always)]
pub(crate) unsafe fn hadd64_pub(v: __m256i) -> u64 {
    unsafe { hadd64(v) }
}

#[inline(always)]
unsafe fn hadd64(v: __m256i) -> u64 {
    unsafe {
        let lo128 = _mm256_castsi256_si128(v);
        let hi128 = _mm256_extracti128_si256::<1>(v);
        let sum2 = _mm_add_epi64(lo128, hi128); // [l0+l2, l1+l3]
        let sum2h = _mm_unpackhi_epi64(sum2, sum2); // [l1+l3, l1+l3]
        let sum1 = _mm_add_epi64(sum2, sum2h); // [total, total]
        _mm_cvtsi128_si64(sum1) as u64
    }
}

/// Vectorized CRT weighted accumulation: `v = Σ_k t[k] * qm[k]` in u128.
///
/// Decomposes each `qm[k]` into three 32-bit limbs (HI/MID/LO) and uses
/// `_mm256_mul_epu32` for all four lane products simultaneously.
/// Replaces 8 scalar `MUL r64` + 1 `vmovdqu` store with 3 × `_mm256_mul_epu32`.
///
/// **Bounds** (ensures no u64 overflow in horizontal sums):
/// - `s_hi  < 4 · (Q_max - 1) · 2^26 < 2^59`  → fits in u64
/// - `s_mid < 4 · (Q_max - 1) · 2^32 < 2^64`  → fits in u64
/// - `s_lo  < 4 · (Q_max - 1) · 2^32 < 2^64`  → fits in u64
/// - `v = s_hi * 2^64 + s_mid * 2^32 + s_lo < 4 · total_q < 2^122` → fits in u128
///
/// # Safety
///
/// `t` must hold values `< Q[k]` in each lane (output of [`reduce_b_and_apply_crt`]).
/// Caller must ensure AVX-512F support.
#[inline(always)]
pub(crate) unsafe fn crt_accumulate_avx512(t: __m256i, qm_hi: __m256i, qm_mid: __m256i, qm_lo: __m256i) -> u128 {
    unsafe {
        let p_hi = _mm256_mul_epu32(t, qm_hi); // t[k] * QM_HI[k]  < 2^57/lane
        let p_mid = _mm256_mul_epu32(t, qm_mid); // t[k] * QM_MID[k] < 2^62/lane
        let p_lo = _mm256_mul_epu32(t, qm_lo); // t[k] * QM_LO[k]  < 2^62/lane

        let s_hi = hadd64(p_hi); // < 4 · 2^57 = 2^59          ✓
        let s_mid = hadd64(p_mid); // < 4 · (Q-1) · 2^32 < 2^64  ✓
        let s_lo = hadd64(p_lo); // < 4 · (Q-1) · 2^32 < 2^64  ✓

        // v = s_hi·2^64 + s_mid·2^32 + s_lo  (no u128 overflow: v < 4·total_q < 2^122)
        ((s_hi as u128) << 64) + ((s_mid as u128) << 32) + (s_lo as u128)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b_from_znx64_avx512
// ─────────────────────────────────────────────────────────────────────────────

/// AVX-512F port of `b_from_znx64_ref`: convert `i64` coefficients to q120b.
///
/// For each coefficient `x[j]`:
/// - Strips the sign bit to get `xl = x[j] as u64 & i64::MAX`.
/// - For negative inputs, adds `oq[k]` per prime so the result is congruent to `x[j]` mod `Q[k]`.
///
/// Processes two coefficients per loop iteration, writing one `__m512i` (8 × u64)
/// to `res`. The odd-coefficient tail uses a single 256-bit iteration.
///
/// # Safety
///
/// Caller must ensure AVX-512F support. `res.len() >= 4 * nn`, `x.len() >= nn`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn b_from_znx64_avx512(nn: usize, res: &mut [u64], x: &[i64]) {
    assert!(
        res.len() >= 4 * nn,
        "b_from_znx64_avx512: res.len()={} < 4*nn={}",
        res.len(),
        4 * nn
    );
    assert!(x.len() >= nn, "b_from_znx64_avx512: x.len()={} < nn={}", x.len(), nn);
    unsafe {
        let oq_vec_512 = bcast_quad(OQ.as_ptr());
        let i64_max_512 = _mm512_set1_epi64(i64::MAX);
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;

        let pairs = nn / 2;
        let mut idx = 0usize;
        for _ in 0..pairs {
            let x0 = *x.get_unchecked(idx);
            let x1 = *x.get_unchecked(idx + 1);
            // 8 lanes: [x0,x0,x0,x0, x1,x1,x1,x1]
            let xv = _mm512_set_epi64(x1, x1, x1, x1, x0, x0, x0, x0);
            let xl = _mm512_and_si512(xv, i64_max_512);
            // For each lane, add oq[k] iff xv < 0 (sign-bit set in original i64).
            // Mask form: _mm512_cmpgt_epi64_mask(0, xv) returns 1 where xv < 0 (signed).
            // Use that mask to selectively add oq.
            let sign_mask = _mm512_cmpgt_epi64_mask(_mm512_setzero_si512(), xv);
            _mm512_storeu_si512(r_ptr, _mm512_mask_add_epi64(xl, sign_mask, xl, oq_vec_512));
            r_ptr = r_ptr.add(1);
            idx += 2;
        }
        if nn & 1 != 0 {
            let oq_vec = _mm256_loadu_si256(OQ.as_ptr() as *const __m256i);
            let i64_max = _mm256_set1_epi64x(i64::MAX);
            let zero = _mm256_setzero_si256();
            let xval = *x.get_unchecked(idx);
            let xv = _mm256_set1_epi64x(xval);
            let xl = _mm256_and_si256(xv, i64_max);
            let sign = _mm256_cmpgt_epi64(zero, xv);
            let add = _mm256_and_si256(sign, oq_vec);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(xl, add));
        }
    }
}

/// AVX-512F variant of `b_from_znx64_masked_ref`: mask coefficients before q120b conversion.
///
/// Caller must ensure AVX-512F support. `res.len() >= 4 * nn`, `x.len() >= nn`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn b_from_znx64_masked_avx512(nn: usize, res: &mut [u64], x: &[i64], mask: i64) {
    assert!(
        res.len() >= 4 * nn,
        "b_from_znx64_masked_avx512: res.len()={} < 4*nn={}",
        res.len(),
        4 * nn
    );
    assert!(x.len() >= nn, "b_from_znx64_masked_avx512: x.len()={} < nn={}", x.len(), nn);
    unsafe {
        let oq_vec_512 = bcast_quad(OQ.as_ptr());
        let i64_max_512 = _mm512_set1_epi64(i64::MAX);
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;

        let pairs = nn / 2;
        let mut idx = 0usize;
        for _ in 0..pairs {
            let x0 = *x.get_unchecked(idx) & mask;
            let x1 = *x.get_unchecked(idx + 1) & mask;
            let xv = _mm512_set_epi64(x1, x1, x1, x1, x0, x0, x0, x0);
            let xl = _mm512_and_si512(xv, i64_max_512);
            let sign_mask = _mm512_cmpgt_epi64_mask(_mm512_setzero_si512(), xv);
            _mm512_storeu_si512(r_ptr, _mm512_mask_add_epi64(xl, sign_mask, xl, oq_vec_512));
            r_ptr = r_ptr.add(1);
            idx += 2;
        }
        if nn & 1 != 0 {
            let oq_vec = _mm256_loadu_si256(OQ.as_ptr() as *const __m256i);
            let i64_max = _mm256_set1_epi64x(i64::MAX);
            let zero = _mm256_setzero_si256();
            let xval = *x.get_unchecked(idx) & mask;
            let xv = _mm256_set1_epi64x(xval);
            let xl = _mm256_and_si256(xv, i64_max);
            let sign = _mm256_cmpgt_epi64(zero, xv);
            let add = _mm256_and_si256(sign, oq_vec);
            _mm256_storeu_si256(r_ptr as *mut __m256i, _mm256_add_epi64(xl, add));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// c_from_b_avx512
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce a single q120b `__m256i` to the canonical residue in `[0, Q[k])` for each prime.
///
/// Input `x` holds values in `[0, Q[k] << 33)`, so `x < 2^63`.
/// Returns the residue in the lower 32 bits of each 64-bit lane.
#[inline(always)]
pub(crate) unsafe fn reduce_b_to_canonical(x: __m256i, q: __m256i, mu: __m256i, pow32: __m256i) -> __m256i {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // x_hi = x >> 32 < 2 * Q[k] (since x < Q << 33)
        let x_hi = _mm256_srli_epi64::<32>(x);
        let x_lo = _mm256_and_si256(x, mask32);
        // Reduce x_hi to [0, Q) with one conditional subtract
        let x_hi_r = cond_sub(x_hi, q);
        // tmp = x_hi_r * pow32 + x_lo  (<  Q * Q + 2^32 < 2^60 + 2^32 < 2^61)
        let tmp = _mm256_add_epi64(_mm256_mul_epu32(x_hi_r, pow32), x_lo);
        // Barrett-reduce tmp to [0, Q)
        barrett_reduce(tmp, q, mu)
    }
}

/// Fused q120b reduce + CRT multiply in a single Barrett pass.
///
/// Computes `t[k] = (x[k] * CRT_CST[k]) mod Q[k]` for all four prime lanes simultaneously,
/// starting from a q120b value `x[k] < Q[k] << 33`, using **one** Barrett reduction instead
/// of the two-step `reduce_b_to_canonical` + `barrett(x * CRT)` sequence.
///
/// The key identity is:
/// ```text
/// x * CRT ≡ x_hi_r * POW32_CRT + x_lo_hi * POW16_CRT + x_lo_lo * CRT  (mod Q)
/// ```
/// where `x = x_hi * 2^32 + x_lo_hi * 2^16 + x_lo_lo`, `x_hi_r = cond_sub(x_hi, Q)`.
///
/// The three-part split keeps every sub-product below `2^61`:
/// - `x_hi_r * POW32_CRT < Q^2 < 2^60`
/// - `x_lo_hi * POW16_CRT < 2^16 * Q < 2^46`
/// - `x_lo_lo * CRT       < 2^16 * Q < 2^46`
/// - `sum < 2^60 + 2^47 < 2^61` ✓
///
/// Saves one Barrett pass (and two conditional subtracts) vs the two-step approach.
#[inline(always)]
pub(crate) unsafe fn reduce_b_and_apply_crt(
    x: __m256i,
    q: __m256i,
    mu: __m256i,
    pow32_crt: __m256i,
    pow16_crt: __m256i,
    crt: __m256i,
) -> __m256i {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mask16 = _mm256_set1_epi64x(0xFFFF_i64);
        // x_hi = x >> 32 < 2*Q  (x < Q << 33)
        let x_hi = _mm256_srli_epi64::<32>(x);
        // x_hi_r < Q after one conditional subtract
        let x_hi_r = cond_sub(x_hi, q);
        // x_lo_hi and x_lo_lo split the lower 32 bits at bit 16
        let x_lo = _mm256_and_si256(x, mask32);
        let x_lo_hi = _mm256_srli_epi64::<16>(x_lo);
        let x_lo_lo = _mm256_and_si256(x_lo, mask16);
        // tmp = x_hi_r * POW32_CRT + x_lo_hi * POW16_CRT + x_lo_lo * CRT < 2^61
        let p1 = _mm256_mul_epu32(x_hi_r, pow32_crt);
        let p2 = _mm256_mul_epu32(x_lo_hi, pow16_crt);
        let p3 = _mm256_mul_epu32(x_lo_lo, crt);
        let tmp = _mm256_add_epi64(_mm256_add_epi64(p1, p2), p3);
        barrett_reduce(tmp, q, mu)
    }
}

/// AVX-512F port of `c_from_b_ref`: convert q120b to q120c.
///
/// For each of `nn` ring elements, reads one `__m256i` (4 × u64, q120b layout) and writes
/// one `__m256i` (8 × u32, q120c layout `[r[0], r_shift[0], ..., r[3], r_shift[3]]`).
/// The 512-bit main loop processes two coefficients per iteration; a 256-bit
/// tail covers the odd-coefficient case.
///
/// # Safety
///
/// Caller must ensure AVX-512F support. `res.len() >= 8 * nn`, `a.len() >= 4 * nn`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn c_from_b_avx512(nn: usize, res: &mut [u32], a: &[u64]) {
    assert!(
        res.len() >= 8 * nn,
        "c_from_b_avx512: res.len()={} < 8*nn={}",
        res.len(),
        8 * nn
    );
    assert!(a.len() >= 4 * nn, "c_from_b_avx512: a.len()={} < 4*nn={}", a.len(), 4 * nn);
    unsafe {
        let q_512 = bcast_quad(Q_VEC.as_ptr());
        let mu_512 = bcast_quad(BARRETT_MU.as_ptr());
        let pow32_512 = bcast_quad(POW32.as_ptr());
        let mut a_ptr = a.as_ptr() as *const __m512i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m512i;

        let pairs = nn / 2;
        for _ in 0..pairs {
            let xv = _mm512_loadu_si512(a_ptr);
            let r = reduce_b_to_canonical_512(xv, q_512, mu_512, pow32_512);
            let r_shift = barrett_reduce_512(_mm512_mul_epu32(r, pow32_512), q_512, mu_512);
            let packed = _mm512_or_si512(r, _mm512_slli_epi64::<32>(r_shift));
            _mm512_storeu_si512(r_ptr, packed);
            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
        if nn & 1 != 0 {
            let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
            let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
            let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
            let xv = _mm256_loadu_si256(a_ptr as *const __m256i);
            let r = reduce_b_to_canonical(xv, q, mu, pow32);
            let r_shift = barrett_reduce(_mm256_mul_epu32(r, pow32), q, mu);
            let packed = _mm256_or_si256(r, _mm256_slli_epi64::<32>(r_shift));
            _mm256_storeu_si256(r_ptr as *mut __m256i, packed);
        }
    }
}

/// AVX-512F pack for a row range of q120b x2-blocks.
///
/// `a` is a column-start q120b slice with row stride `row_stride` (in `u64` units).
/// For each row, block `blk` is reduced to canonical residues and written to `dst`
/// in the x2 q120b/u32 layout expected by BBC kernels.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pack_left_1blk_x2_avx512(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);

    // Each row reads 2 q120b (16 u64) and writes 16 u32 reduced residues. Both q120b's
    // share the same per-prime constants, so we pair-pack them into one __m512i.
    unsafe {
        let q_512 = bcast_quad(Q_VEC.as_ptr());
        let mu_512 = bcast_quad(BARRETT_MU.as_ptr());
        let pow32_512 = bcast_quad(POW32.as_ptr());
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;

        for _ in 0..row_count {
            let a01 = _mm512_loadu_si512(a_ptr);
            let r01 = reduce_b_to_canonical_512(a01, q_512, mu_512, pow32_512);
            _mm512_storeu_si512(dst_ptr, r01);

            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// AVX-512F pack for a row range of q120c x2-blocks in reversed row order.
///
/// `a` is a column-start q120c slice with row stride `row_stride` (in `u32` units).
/// For each row, block `blk` is copied to `dst` in reversed row order so convolution
/// windows can consume contiguous slices directly.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pack_right_1blk_x2_avx512(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);

    // Pure 16-u32 copy per row in reversed row order. Each row is one 512-bit transfer.
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk) as *const __m512i;

        for _ in 0..row_count {
            _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(a_ptr));
            a_ptr = (a_ptr as *const u32).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// AVX-512F pairwise pack for a row range of q120b x2-blocks.
///
/// `a` and `b` are column-start q120b slices with row stride `row_stride` (in `u64` units).
/// For each row, block `blk` is reduced to canonical residues, summed mod `Q`,
/// and written to `dst` in the x2 q120b/u32 layout expected by BBC kernels.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pairwise_pack_left_1blk_x2_avx512(
    dst: &mut [u32],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);

    // Each row's 2 q120b's are pair-packed into one __m512i; reduce + sum mod Q in 8 lanes.
    unsafe {
        let q_512 = bcast_quad(Q_VEC.as_ptr());
        let mu_512 = bcast_quad(BARRETT_MU.as_ptr());
        let pow32_512 = bcast_quad(POW32.as_ptr());
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(8 * blk) as *const __m512i;

        for _ in 0..row_count {
            let av = _mm512_loadu_si512(a_ptr);
            let bv = _mm512_loadu_si512(b_ptr);
            let r = reduce_b_to_canonical_512(av, q_512, mu_512, pow32_512);
            let s = reduce_b_to_canonical_512(bv, q_512, mu_512, pow32_512);
            _mm512_storeu_si512(dst_ptr, cond_sub_512(_mm512_add_epi64(r, s), q_512));

            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// AVX-512F pairwise pack for a row range of q120c x2-blocks.
///
/// `a` and `b` are column-start q120c slices with row stride `row_stride` (in `u32` units).
/// For each row, block `blk` is written to `dst` in reversed row order so convolution windows
/// can consume contiguous slices directly.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pairwise_pack_right_1blk_x2_avx512(
    dst: &mut [u32],
    a: &[u32],
    b: &[u32],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);

    // 16-u32 lanewise add per row (q120c element-wise sum, no mod Q reduction needed).
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk) as *const __m512i;

        for _ in 0..row_count {
            _mm512_storeu_si512(
                dst_ptr,
                _mm512_add_epi32(_mm512_loadu_si512(a_ptr), _mm512_loadu_si512(b_ptr)),
            );
            a_ptr = (a_ptr as *const u32).sub(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u32).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// vec_mat1col_product_bbb_avx512
// ─────────────────────────────────────────────────────────────────────────────

/// AVX-512F port of `vec_mat1col_product_bbb_ref`: q120b × q120b → q120b dot product.
///
/// Computes `res = Σᵢ x[i] · y[i]` in q120b format for `i ∈ 0..ell`. Each
/// element is one `__m256i` (4 × u64, one u64 per prime). The 512-bit main
/// loop accumulates two elements per iteration; the two halves are folded
/// (per-prime added) before the final `BbbMeta` reduction.
///
/// Uses a four-bin accumulation scheme (`s1`–`s4`) matching the scalar reference.
///
/// # Safety
///
/// Caller must ensure AVX-512F support. `res.len() >= 4`, `x.len() >= 4 * ell`, `y.len() >= 4 * ell`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn vec_mat1col_product_bbb_avx512(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    assert!(res.len() >= 4, "vec_mat1col_product_bbb_avx512: res.len()={} < 4", res.len());
    assert!(
        x.len() >= 4 * ell,
        "vec_mat1col_product_bbb_avx512: x.len()={} < 4*ell={}",
        x.len(),
        4 * ell
    );
    assert!(
        y.len() >= 4 * ell,
        "vec_mat1col_product_bbb_avx512: y.len()={} < 4*ell={}",
        y.len(),
        4 * ell
    );
    // Pair-pack inner accumulation: process 2 elements per __m512i iteration.
    // Both halves accumulate into separate 4-prime bins; the two halves are folded
    // (per-prime added) just before the final BbbMeta reduction.
    unsafe {
        let mask32_512 = _mm512_set1_epi64(u32::MAX as i64);
        let mut s1 = _mm512_setzero_si512();
        let mut s2 = _mm512_setzero_si512();
        let mut s3 = _mm512_setzero_si512();
        let mut s4 = _mm512_setzero_si512();

        let mut x_ptr = x.as_ptr() as *const __m512i;
        let mut y_ptr = y.as_ptr() as *const __m512i;

        let pairs = ell / 2;
        for _ in 0..pairs {
            let xv = _mm512_loadu_si512(x_ptr);
            let xl = _mm512_and_si512(xv, mask32_512);
            let xh = _mm512_srli_epi64::<32>(xv);

            let yv = _mm512_loadu_si512(y_ptr);
            let yl = _mm512_and_si512(yv, mask32_512);
            let yh = _mm512_srli_epi64::<32>(yv);

            let a = _mm512_mul_epu32(xl, yl);
            let b = _mm512_mul_epu32(xl, yh);
            let c = _mm512_mul_epu32(xh, yl);
            let d = _mm512_mul_epu32(xh, yh);

            s1 = _mm512_add_epi64(s1, _mm512_and_si512(a, mask32_512));
            s2 = _mm512_add_epi64(s2, _mm512_srli_epi64::<32>(a));
            s2 = _mm512_add_epi64(s2, _mm512_and_si512(b, mask32_512));
            s2 = _mm512_add_epi64(s2, _mm512_and_si512(c, mask32_512));
            s3 = _mm512_add_epi64(s3, _mm512_srli_epi64::<32>(b));
            s3 = _mm512_add_epi64(s3, _mm512_srli_epi64::<32>(c));
            s3 = _mm512_add_epi64(s3, _mm512_and_si512(d, mask32_512));
            s4 = _mm512_add_epi64(s4, _mm512_srli_epi64::<32>(d));

            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        // Fold the two 256-bit halves: per-prime sum across the pair-packed accumulators.
        let s1_lo = _mm512_extracti64x4_epi64::<0>(s1);
        let s1_hi = _mm512_extracti64x4_epi64::<1>(s1);
        let mut s1 = _mm256_add_epi64(s1_lo, s1_hi);
        let s2_lo = _mm512_extracti64x4_epi64::<0>(s2);
        let s2_hi = _mm512_extracti64x4_epi64::<1>(s2);
        let mut s2 = _mm256_add_epi64(s2_lo, s2_hi);
        let s3_lo = _mm512_extracti64x4_epi64::<0>(s3);
        let s3_hi = _mm512_extracti64x4_epi64::<1>(s3);
        let mut s3 = _mm256_add_epi64(s3_lo, s3_hi);
        let s4_lo = _mm512_extracti64x4_epi64::<0>(s4);
        let s4_hi = _mm512_extracti64x4_epi64::<1>(s4);
        let mut s4 = _mm256_add_epi64(s4_lo, s4_hi);

        // Tail: handle the odd element (if ell is odd) using the 256-bit path.
        if ell & 1 != 0 {
            let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
            let xv = _mm256_loadu_si256(x_ptr as *const __m256i);
            let xl = _mm256_and_si256(xv, mask32);
            let xh = _mm256_srli_epi64::<32>(xv);
            let yv = _mm256_loadu_si256(y_ptr as *const __m256i);
            let yl = _mm256_and_si256(yv, mask32);
            let yh = _mm256_srli_epi64::<32>(yv);

            let a = _mm256_mul_epu32(xl, yl);
            let b = _mm256_mul_epu32(xl, yh);
            let c = _mm256_mul_epu32(xh, yl);
            let d = _mm256_mul_epu32(xh, yh);

            s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(a));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(b, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(c, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(b));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(c));
            s3 = _mm256_add_epi64(s3, _mm256_and_si256(d, mask32));
            s4 = _mm256_add_epi64(s4, _mm256_srli_epi64::<32>(d));
        }

        // Final reduction using BbbMeta constants (4-lane / per-prime).
        let h2 = meta.h;
        let mask_h2 = _mm256_set1_epi64x(((1u64 << h2) - 1) as i64);
        let h2_cnt = _mm_cvtsi64_si128(h2 as i64);
        let s1h_pow = _mm256_set1_epi64x(meta.s1h_pow_red as i64);
        let s2l_pow = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);
        let s3l_pow = _mm256_loadu_si256(meta.s3l_pow_red.as_ptr() as *const __m256i);
        let s3h_pow = _mm256_loadu_si256(meta.s3h_pow_red.as_ptr() as *const __m256i);
        let s4l_pow = _mm256_loadu_si256(meta.s4l_pow_red.as_ptr() as *const __m256i);
        let s4h_pow = _mm256_loadu_si256(meta.s4h_pow_red.as_ptr() as *const __m256i);

        let s1l = _mm256_and_si256(s1, mask_h2);
        let s1h = _mm256_srl_epi64(s1, h2_cnt);
        let s2l = _mm256_and_si256(s2, mask_h2);
        let s2h = _mm256_srl_epi64(s2, h2_cnt);
        let s3l = _mm256_and_si256(s3, mask_h2);
        let s3h = _mm256_srl_epi64(s3, h2_cnt);
        let s4l = _mm256_and_si256(s4, mask_h2);
        let s4h = _mm256_srl_epi64(s4, h2_cnt);

        let mut t = s1l;
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s1h, s1h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s2l, s2l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s2h, s2h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s3l, s3l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s3h, s3h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s4l, s4l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s4h, s4h_pow));

        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b_to_znx128_avx512
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid AVX-512F / scalar CRT reconstruction: q120b → i128 coefficients.
///
/// For each of `nn` ring elements:
/// - **AVX-512F**: Computes `t[k] = (x[4*j+k] % Q[k] * CRT_CST[k]) % Q[k]` for k=0..3.
/// - **Scalar**: Accumulates `tmp = Σ_k t[k] * (Q/Q[k])` in i128, reduces mod `total_Q`,
///   and applies a symmetric lift to `(-total_Q/2, total_Q/2]`.
///
/// The 512-bit main loop runs the Barrett pass and the 32-bit-limb CRT
/// products on two coefficients in parallel; the per-coefficient scalar fold
/// then runs sequentially over the two halves.
///
/// # Safety
///
/// Caller must ensure AVX-512F support. `res.len() >= nn`, `a.len() >= 4 * nn`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn b_to_znx128_avx512(nn: usize, res: &mut [i128], a: &[u64]) {
    assert!(res.len() >= nn, "b_to_znx128_avx512: res.len()={} < nn={}", res.len(), nn);
    assert!(a.len() >= 4 * nn, "b_to_znx128_avx512: a.len()={} < 4*nn={}", a.len(), 4 * nn);
    let half_q: u128 = TOTAL_Q.div_ceil(2);

    // Pair-packed (2-coefficient) AVX-512 path: vectorize the Barrett pass and the
    // 32-bit-limb CRT product across 8 lanes. Per-coefficient horizontal sum + scalar
    // modular fold + sign lift then runs sequentially over the two halves.
    unsafe {
        let q_512 = bcast_quad(Q_VEC.as_ptr());
        let mu_512 = bcast_quad(BARRETT_MU.as_ptr());
        let pow32_crt_512 = bcast_quad(POW32_CRT.as_ptr());
        let pow16_crt_512 = bcast_quad(POW16_CRT.as_ptr());
        let crt_512 = bcast_quad(CRT_VEC.as_ptr());
        let qm_hi_512 = bcast_quad(QM_HI.as_ptr());
        let qm_mid_512 = bcast_quad(QM_MID.as_ptr());
        let qm_lo_512 = bcast_quad(QM_LO.as_ptr());

        let mut a_ptr = a.as_ptr() as *const __m512i;

        let pairs = nn / 2;
        let mut idx = 0usize;
        for _ in 0..pairs {
            let xv = _mm512_loadu_si512(a_ptr);
            // Fused reduce + CRT mul across 8 lanes (2 coefficients).
            let t = reduce_b_and_apply_crt_512(xv, q_512, mu_512, pow32_crt_512, pow16_crt_512, crt_512);
            // Per-limb partial products (8 lanes each).
            let p_hi = _mm512_mul_epu32(t, qm_hi_512);
            let p_mid = _mm512_mul_epu32(t, qm_mid_512);
            let p_lo = _mm512_mul_epu32(t, qm_lo_512);
            // Split halves and horizontal-sum each → (s_hi, s_mid, s_lo) per coefficient.
            let p_hi_a = _mm512_extracti64x4_epi64::<0>(p_hi);
            let p_hi_b = _mm512_extracti64x4_epi64::<1>(p_hi);
            let p_mid_a = _mm512_extracti64x4_epi64::<0>(p_mid);
            let p_mid_b = _mm512_extracti64x4_epi64::<1>(p_mid);
            let p_lo_a = _mm512_extracti64x4_epi64::<0>(p_lo);
            let p_lo_b = _mm512_extracti64x4_epi64::<1>(p_lo);

            for (k, (p_hi_h, (p_mid_h, p_lo_h))) in [(p_hi_a, (p_mid_a, p_lo_a)), (p_hi_b, (p_mid_b, p_lo_b))]
                .into_iter()
                .enumerate()
            {
                let s_hi = hadd64(p_hi_h);
                let s_mid = hadd64(p_mid_h);
                let s_lo = hadd64(p_lo_h);
                let mut v: u128 = ((s_hi as u128) << 64) + ((s_mid as u128) << 32) + (s_lo as u128);
                let q_approx = (v >> 120) as usize;
                v -= TOTAL_Q_MULT[q_approx];
                if v >= TOTAL_Q {
                    v -= TOTAL_Q;
                }
                let signed = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                *res.get_unchecked_mut(idx + k) = signed;
            }
            a_ptr = a_ptr.add(1);
            idx += 2;
        }

        if nn & 1 != 0 {
            let q_vec = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
            let mu_vec = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
            let pow32_crt_vec = _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i);
            let pow16_crt_vec = _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i);
            let crt_vec = _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i);
            let qm_hi_vec = _mm256_loadu_si256(QM_HI.as_ptr() as *const __m256i);
            let qm_mid_vec = _mm256_loadu_si256(QM_MID.as_ptr() as *const __m256i);
            let qm_lo_vec = _mm256_loadu_si256(QM_LO.as_ptr() as *const __m256i);
            let xv = _mm256_loadu_si256(a_ptr as *const __m256i);
            let t = reduce_b_and_apply_crt(xv, q_vec, mu_vec, pow32_crt_vec, pow16_crt_vec, crt_vec);
            let mut v = crt_accumulate_avx512(t, qm_hi_vec, qm_mid_vec, qm_lo_vec);
            let q_approx = (v >> 120) as usize;
            v -= TOTAL_Q_MULT[q_approx];
            if v >= TOTAL_Q {
                v -= TOTAL_Q;
            }
            *res.get_unchecked_mut(idx) = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt120::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref, c_from_b_ref},
        mat_vec::{BbbMeta, vec_mat1col_product_bbb_ref},
        primes::Primes30,
    };

    /// AVX-512F `b_from_znx64` matches reference for arbitrary i64 inputs.
    #[test]
    fn b_from_znx64_avx2_vs_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i * 17 - 500).collect();

        let mut res_avx = vec![0u64; 4 * n];
        let mut res_ref = vec![0u64; 4 * n];

        unsafe { b_from_znx64_avx512(n, &mut res_avx, &coeffs) };
        b_from_znx64_ref::<Primes30>(n, &mut res_ref, &coeffs);

        assert_eq!(res_avx, res_ref, "b_from_znx64: AVX-512F vs ref mismatch");
    }

    /// AVX-512F `c_from_b` (Barrett reduction to Montgomery u32) matches reference.
    #[test]
    fn c_from_b_avx2_vs_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i * 11 + 3).collect();

        let mut b = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut b, &coeffs);

        let mut res_avx = vec![0u32; 8 * n];
        let mut res_ref = vec![0u32; 8 * n];

        unsafe { c_from_b_avx512(n, &mut res_avx, &b) };
        c_from_b_ref::<Primes30>(n, &mut res_ref, &b);

        assert_eq!(res_avx, res_ref, "c_from_b: AVX-512F vs ref mismatch");
    }

    /// AVX-512F `vec_mat1col_product_bbb` matches reference.
    #[test]
    fn vec_mat1col_product_bbb_avx2_vs_ref() {
        let ell = 16usize;
        let n = 64usize;
        let meta = BbbMeta::<Primes30>::new();

        // Build two q120b matrices (ell * 4*n u64 values)
        let x_i64: Vec<i64> = (0..ell * n).map(|i| (i as i64 * 7 + 1) % 100).collect();
        let y_i64: Vec<i64> = (0..ell * n).map(|i| (i as i64 * 13 + 2) % 100).collect();

        let mut x = vec![0u64; 4 * ell * n];
        let mut y = vec![0u64; 4 * ell * n];
        b_from_znx64_ref::<Primes30>(ell * n, &mut x, &x_i64);
        b_from_znx64_ref::<Primes30>(ell * n, &mut y, &y_i64);

        let mut res_avx = vec![0u64; 4 * n];
        let mut res_ref = vec![0u64; 4 * n];

        unsafe { vec_mat1col_product_bbb_avx512(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat1col_product_bbb_ref::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_bbb: AVX-512F vs ref mismatch");
    }

    /// Fused `reduce_b_and_apply_crt` matches two-step `reduce_b_to_canonical` + barrett.
    #[test]
    fn reduce_b_and_apply_crt_vs_two_step() {
        use poulpy_cpu_ref::reference::ntt120::arithmetic::b_from_znx64_ref;
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i * 5 - 160).collect();
        let mut b = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut b, &coeffs);

        let q = unsafe { _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i) };
        let mu = unsafe { _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i) };
        let pow32 = unsafe { _mm256_loadu_si256(POW32.as_ptr() as *const __m256i) };
        let crt = unsafe { _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i) };
        let pow32_crt = unsafe { _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i) };
        let pow16_crt = unsafe { _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i) };

        for j in 0..n {
            let xv = unsafe { _mm256_loadu_si256(b[4 * j..].as_ptr() as *const __m256i) };
            let mut two_step = [0u64; 4];
            let mut fused = [0u64; 4];
            unsafe {
                let xk = reduce_b_to_canonical(xv, q, mu, pow32);
                let t = barrett_reduce(_mm256_mul_epu32(xk, crt), q, mu);
                _mm256_storeu_si256(two_step.as_mut_ptr() as *mut __m256i, t);
                let t2 = reduce_b_and_apply_crt(xv, q, mu, pow32_crt, pow16_crt, crt);
                _mm256_storeu_si256(fused.as_mut_ptr() as *mut __m256i, t2);
            }
            assert_eq!(fused, two_step, "reduce_b_and_apply_crt mismatch at j={j}");
        }
    }

    /// AVX-512F `b_to_znx128` matches reference for valid q120b input.
    #[test]
    fn b_to_znx128_avx2_vs_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i * 5 - 160).collect();

        let mut b = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut b, &coeffs);

        let mut res_avx = vec![0i128; n];
        let mut res_ref = vec![0i128; n];

        unsafe { b_to_znx128_avx512(n, &mut res_avx, &b) };
        b_to_znx128_ref::<Primes30>(n, &mut res_ref, &b);

        assert_eq!(res_avx, res_ref, "b_to_znx128: AVX-512F vs ref mismatch");
    }
}
