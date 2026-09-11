//! Raw AVX512-IFMA forward and inverse NTT kernels, 8 u64 lanes per `__m512i`.
//!
//! Forward: Cooley-Tukey, natural-order input -> bit-reversed output. Inverse:
//! Gentleman-Sande, bit-reversed -> natural, with the `1/n` scale folded in.
//! Lazy output ranges are `[0, 4q)` forward and `[0, 2q)` inverse. Canonical
//! forward output adds a final pass reducing to `[0, q)`. SIMD twiddles are
//! reconstructed from their precomputed Harvey quotients.

// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been adapted from the Intel HEXL
// library (https://github.com/intel/hexl), which is licensed under the
// Apache License, Version 2.0.
//
// Unlike the spqlios-arithmetic ports, this is not a 1-to-1 port: the
// kernels were reworked for Poulpy's three-prime CRT layout.
//
// Both Poulpy and HEXL are distributed under the terms of the Apache
// License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd52hi_epu64,
    _mm256_madd52lo_epu64, _mm256_min_epu64, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm512_add_epi64, _mm512_and_si512, _mm512_castsi128_si512, _mm512_castsi256_si512, _mm512_extracti64x4_epi64,
    _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_blend_epi64, _mm512_min_epu64,
    _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi64,
};

use crate::ntt3x42_ifma::{
    primes::PrimeSetNtt3x42Ifma,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv, cond_sub_2q, harvey_modmul},
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD arithmetic primitives
// ──────────────────────────────────────────────────────────────────────────────

/// Conditional subtract of `q2`: if x >= q2 (unsigned), return x - q2, else x.
///
/// Uses `min(x, x − q2 mod 2^64) == x − q2` when `x ≥ q2`, `== x` otherwise.
#[inline]
#[target_feature(enable = "avx512vl")]
pub(crate) unsafe fn cond_sub_2q_si256(x: __m256i, q2: __m256i) -> __m256i {
    let diff = _mm256_sub_epi64(x, q2);
    _mm256_min_epu64(x, diff)
}

/// Harvey modular multiply — 4 lanes.
///
/// Input: `a` up to `8q` (under lazy reduction), `omega ∈ [0, q)`. Output:
/// `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`. Only the low 52 bits of `a·ω` and
/// `qhat·q` are needed; the final mask to 52 bits handles the borrow.
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
/// Accumulate with `-q` modulo 2^52 to fold the subtraction into IFMA.
#[inline]
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn harvey_modmul_si512(a: __m512i, omega: __m512i, omega_quot: __m512i, q: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let mask52 = _mm512_set1_epi64((1i64 << 52) - 1);
    let qhat = _mm512_madd52hi_epu64(zero, a, omega_quot);
    let prod_lo52 = _mm512_madd52lo_epu64(zero, a, omega);
    let neg_q = _mm512_sub_epi64(zero, q);
    let reduced = _mm512_madd52lo_epu64(prod_lo52, qhat, neg_q);
    _mm512_and_si512(reduced, mask52)
}

// For odd q and 0 < w < q < 2^52, floor(floor(w * 2^52 / q) * q / 2^52) = w - 1.
#[inline]
#[target_feature(enable = "avx512ifma")]
unsafe fn recover_root_si512(quotient: __m512i, q: __m512i) -> __m512i {
    _mm512_madd52hi_epu64(_mm512_set1_epi64(1), quotient, q)
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT (Cooley-Tukey, natural -> bit-reversed)
// ──────────────────────────────────────────────────────────────────────────────

// Sums grow by at most 2q per stage: below 33q < 2^48 at n <= 2^16.
// The final tail narrows them before returning the public [0, 4q) result.
#[inline]
#[target_feature(enable = "avx512ifma,avx512f")]
unsafe fn fwd_butterfly_unreduced(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    w_precon: __m512i,
    q: __m512i,
    q2: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        let t = harvey_modmul_si512(y, w, w_precon, q);
        let x_out = _mm512_add_epi64(x, t);
        let y_out = _mm512_add_epi64(x, _mm512_sub_epi64(q2, t));
        (x_out, y_out)
    }
}

/// Forward distance-1 interleaved load: gathers the even/odd lanes of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 1` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_fwd_interleaved_t1(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let v1 = _mm512_loadu_si512(arg as *const __m512i);
        let v2 = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let perm = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        let v1p = _mm512_permutexvar_epi64(perm, v1);
        let v2p = _mm512_permutexvar_epi64(perm, v2);
        let out1 = _mm512_mask_blend_epi64(0xaa, v1, v2p);
        let out2 = _mm512_mask_blend_epi64(0xaa, v1p, v2);
        (out1, out2)
    }
}

/// Forward distance-2 interleaved load: gathers 2-lane groups of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 2` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_fwd_interleaved_t2(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let v1 = _mm512_loadu_si512(arg as *const __m512i);
        let v2 = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let perm = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        let v1p = _mm512_permutexvar_epi64(perm, v1);
        let v2p = _mm512_permutexvar_epi64(perm, v2);
        let out1 = _mm512_mask_blend_epi64(0xcc, v1, v2p);
        let out2 = _mm512_mask_blend_epi64(0xcc, v1p, v2);
        (out1, out2)
    }
}

/// Forward distance-4 interleaved load: gathers 4-lane halves of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 4` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_fwd_interleaved_t4(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let vperm2 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        let v_7to0 = _mm512_loadu_si512(arg as *const __m512i);
        let v_15to8 = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let perm_hi = _mm512_permutexvar_epi64(vperm2, v_15to8);
        let out1 = _mm512_mask_blend_epi64(0x0f, perm_hi, v_7to0);
        let out2 = _mm512_mask_blend_epi64(0xf0, perm_hi, v_7to0);
        let out2 = _mm512_permutexvar_epi64(vperm2, out2);
        (out1, out2)
    }
}

/// Inverse permutation of [`load_fwd_interleaved_t1`]: scatters the butterfly
/// outputs of a `t = 1` stage back to their natural in-plane positions.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn write_fwd_interleaved_t1(arg1: __m512i, arg2: __m512i, out: *mut u64) {
    unsafe {
        let vperm2 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        let v_x_out = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
        let v_y_out = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);
        let arg2 = _mm512_permutexvar_epi64(vperm2, arg2);
        let perm_lo = _mm512_mask_blend_epi64(0x0f, arg1, arg2);
        let perm_hi = _mm512_mask_blend_epi64(0xf0, arg1, arg2);
        let a1 = _mm512_permutexvar_epi64(v_x_out, perm_hi);
        let a2 = _mm512_permutexvar_epi64(v_y_out, perm_lo);
        _mm512_storeu_si512(out as *mut __m512i, a1);
        _mm512_storeu_si512(out.add(8) as *mut __m512i, a2);
    }
}

/// Largest sub-transform handled directly by the breadth-first base case.
/// Sub-transforms larger than this split depth-first for cache locality.
const BASE_NTT_SIZE: usize = 2048;

#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_broadcast_pair(ptr: *mut u64, t: usize, m: usize, precon: &[u64], wi: usize, q: __m512i, q2: __m512i) {
    unsafe {
        let quarter = t / 2;
        for i in 0..m {
            let wp0 = _mm512_set1_epi64(precon[wi + i] as i64);
            let w0 = recover_root_si512(wp0, q);
            let wp1 = _mm512_set1_epi64(precon[2 * (wi + i)] as i64);
            let w1 = recover_root_si512(wp1, q);
            let wp2 = _mm512_set1_epi64(precon[2 * (wi + i) + 1] as i64);
            let w2 = recover_root_si512(wp2, q);
            let group = ptr.add(i * 2 * t);
            for j in (0..quarter).step_by(8) {
                let p0 = group.add(j) as *mut __m512i;
                let p1 = group.add(j + quarter) as *mut __m512i;
                let p2 = group.add(j + 2 * quarter) as *mut __m512i;
                let p3 = group.add(j + 3 * quarter) as *mut __m512i;
                let a = _mm512_loadu_si512(p0);
                let b = _mm512_loadu_si512(p1);
                let c = _mm512_loadu_si512(p2);
                let d = _mm512_loadu_si512(p3);
                let (a, c) = fwd_butterfly_unreduced(a, c, w0, wp0, q, q2);
                let (b, d) = fwd_butterfly_unreduced(b, d, w0, wp0, q, q2);
                let (a, b) = fwd_butterfly_unreduced(a, b, w1, wp1, q, q2);
                let (c, d) = fwd_butterfly_unreduced(c, d, w2, wp2, q, q2);
                _mm512_storeu_si512(p0, a);
                _mm512_storeu_si512(p1, b);
                _mm512_storeu_si512(p2, c);
                _mm512_storeu_si512(p3, d);
            }
        }
    }
}

/// Breadth-first forward transform of one sub-plane of length `n_sub`.
///
/// `(depth, half)` locate the sub-plane within the depth-first recursion so the
/// shared full-`n` root tables can be indexed: broadcast stages
/// (`t = n_sub/2, …, 8`) start at `W_idx = (m << depth) + half * m` and double
/// `W_idx` per stage; `tail_offset` then maps `W_idx` into the duplicated `tail`
/// block layout (using `N = n_sub << depth`).
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_plane_base(
    ptr: *mut u64,
    n_sub: usize,
    depth: u32,
    half: usize,
    root: &[u64],
    precon: &[u64],
    tail_p: &[u64],
    q: u64,
    q2: u64,
    q_v: __m512i,
    q2_v: __m512i,
) {
    unsafe {
        // Broadcast-twiddle stages: t = n_sub/2, n_sub/4, …, 8.
        let mut t = n_sub / 2;
        let mut m = 1usize;
        let mut w_idx = (m << depth) + half * m;
        while t >= 16 {
            fwd_broadcast_pair(ptr, t, m, precon, w_idx, q_v, q2_v);
            t >>= 2;
            m <<= 2;
            w_idx <<= 2;
        }
        while t >= 8 {
            let mut j1 = 0usize;
            for i in 0..m {
                let w_precon = precon[w_idx + i];
                let w_precon_v = _mm512_set1_epi64(w_precon as i64);
                let w_v = recover_root_si512(w_precon_v, q_v);

                // t/8 independent vector butterflies over (j, j+t).
                let mut j = j1;
                while j < j1 + t {
                    let xp = ptr.add(j) as *mut __m512i;
                    let yp = ptr.add(j + t) as *mut __m512i;
                    let x_in = _mm512_loadu_si512(xp as *const __m512i);
                    let y_in = _mm512_loadu_si512(yp as *const __m512i);
                    let x_red = x_in;
                    let tt = harvey_modmul_si512(y_in, w_v, w_precon_v, q_v);
                    let x_out = _mm512_add_epi64(x_red, tt);
                    let y_out = _mm512_sub_epi64(_mm512_add_epi64(x_red, q2_v), tt);
                    _mm512_storeu_si512(xp, x_out);
                    _mm512_storeu_si512(yp, y_out);
                    j += 8;
                }
                j1 += 2 * t;
            }
            t >>= 1;
            m <<= 1;
            w_idx <<= 1;
        }

        // Map a broadcast `W_idx` into the duplicated `tail` block layout.
        // `N` is the full transform size; the three tail blocks (t = 4, 2, 1)
        // sit at local offsets 0, N/2, N, all reachable as `new_idx - N/8`.
        let big_n = n_sub << depth;
        let tail_offset = |idx: usize| -> usize {
            let new_idx = if idx <= big_n / 8 {
                idx
            } else if idx <= big_n / 4 {
                (idx - big_n / 8) * 4 + big_n / 8
            } else if idx <= big_n / 2 {
                (idx - big_n / 4) * 2 + 5 * big_n / 8
            } else {
                idx + 5 * big_n / 8
            };
            new_idx - big_n / 8
        };

        // Tail stages: t = 4, 2, 1.
        if n_sub >= 16 {
            // t = 4 stage, m = n_sub/8. Distance-4 root block.
            {
                let off = tail_offset(w_idx);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t4(ptr.add(j1));
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let w_v = recover_root_si512(wp_v, q_v);
                    let (vx, vy) = fwd_butterfly_unreduced(vx, vy, w_v, wp_v, q_v, q2_v);
                    _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                    _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                    wp_ptr = wp_ptr.add(8);
                    j1 += 16;
                }
            }

            // t = 2 stage, m = n_sub/4. Distance-2 root block.
            {
                let off = tail_offset(w_idx << 1);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t2(ptr.add(j1));
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let w_v = recover_root_si512(wp_v, q_v);
                    let (vx, vy) = fwd_butterfly_unreduced(vx, vy, w_v, wp_v, q_v, q2_v);
                    _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                    _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                    wp_ptr = wp_ptr.add(8);
                    j1 += 16;
                }
            }

            // t = 1 stage, m = n_sub/2. Distance-1 root block.
            {
                let off = tail_offset(w_idx << 2);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t1(ptr.add(j1));
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let w_v = recover_root_si512(wp_v, q_v);
                    // Reduce by multiplying by one; its low product is already vx.
                    let quotient = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vx, _mm512_set1_epi64(precon[0] as i64));
                    let vx = _mm512_madd52lo_epu64(vx, quotient, _mm512_sub_epi64(_mm512_setzero_si512(), q_v));
                    let vx = _mm512_and_si512(vx, _mm512_set1_epi64((1i64 << 52) - 1));
                    let (vx, vy) = fwd_butterfly_unreduced(vx, vy, w_v, wp_v, q_v, q2_v);
                    write_fwd_interleaved_t1(vx, vy, ptr.add(j1));
                    wp_ptr = wp_ptr.add(8);
                    j1 += 16;
                }
            }
        } else {
            // Scalar tail (n_sub < 16): t = 4, 2, 1. Only reached at depth 0,
            // where W_idx == m, matching the non-duplicated `root` layout.
            while t >= 1 {
                let mut j1 = 0usize;
                for i in 0..m {
                    let w = root[m + i];
                    let w_precon = precon[m + i];
                    for j in j1..j1 + t {
                        let x_in = *ptr.add(j);
                        let y_in = *ptr.add(j + t);
                        let x_red = cond_sub_2q(x_in, q2);
                        let tt = harvey_modmul(y_in, w, w_precon, q);
                        *ptr.add(j) = x_red + tt;
                        *ptr.add(j + t) = x_red + q2 - tt;
                    }
                    j1 += 2 * t;
                }
                if t == 1 {
                    break;
                }
                t >>= 1;
                m <<= 1;
            }
        }
    }
}

/// Depth-first forward transform of one sub-plane of length `n_sub`.
///
/// Large planes pair their top stages and recurse into four children; the
/// remaining split uses two children. Cache-sized planes use the base case.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_plane(
    ptr: *mut u64,
    n_sub: usize,
    depth: u32,
    half: usize,
    root: &[u64],
    precon: &[u64],
    tail_p: &[u64],
    q: u64,
    q2: u64,
    q_v: __m512i,
    q2_v: __m512i,
) {
    unsafe {
        if n_sub <= BASE_NTT_SIZE {
            fwd_plane_base(ptr, n_sub, depth, half, root, precon, tail_p, q, q2, q_v, q2_v);
            return;
        }

        if n_sub >= 4 * BASE_NTT_SIZE {
            let wi = (1usize << depth) + half;
            fwd_broadcast_pair(ptr, n_sub / 2, 1, precon, wi, q_v, q2_v);
            let quarter = n_sub / 4;
            for part in 0..4 {
                fwd_plane(
                    ptr.add(part * quarter),
                    quarter,
                    depth + 2,
                    half * 4 + part,
                    root,
                    precon,
                    tail_p,
                    q,
                    q2,
                    q_v,
                    q2_v,
                );
            }
            return;
        }

        // Top broadcast stage: distance t = n_sub/2, single twiddle group.
        let t = n_sub / 2;
        let w_idx = (1usize << depth) + half;
        let w_precon = precon[w_idx];
        let w_precon_v = _mm512_set1_epi64(w_precon as i64);
        let w_v = recover_root_si512(w_precon_v, q_v);
        let mut j = 0usize;
        while j < t {
            let xp = ptr.add(j) as *mut __m512i;
            let yp = ptr.add(j + t) as *mut __m512i;
            let x_in = _mm512_loadu_si512(xp as *const __m512i);
            let y_in = _mm512_loadu_si512(yp as *const __m512i);
            let x_red = x_in;
            let tt = harvey_modmul_si512(y_in, w_v, w_precon_v, q_v);
            let x_out = _mm512_add_epi64(x_red, tt);
            let y_out = _mm512_sub_epi64(_mm512_add_epi64(x_red, q2_v), tt);
            _mm512_storeu_si512(xp, x_out);
            _mm512_storeu_si512(yp, y_out);
            j += 8;
        }

        // Recurse into the two halves with unreduced sums.
        let half_n = n_sub / 2;
        fwd_plane(ptr, half_n, depth + 1, half * 2, root, precon, tail_p, q, q2, q_v, q2_v);
        fwd_plane(
            ptr.add(half_n),
            half_n,
            depth + 1,
            half * 2 + 1,
            root,
            precon,
            tail_p,
            q,
            q2,
            q_v,
            q2_v,
        );
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn fwd_top2<const N: usize>(ptr: *mut u64, precon: &[u64], q_v: __m512i, q2_v: __m512i) {
    unsafe {
        let quarter = N / 4;
        let wp0 = _mm512_set1_epi64(precon[1] as i64);
        let w0 = recover_root_si512(wp0, q_v);
        let wp1 = _mm512_set1_epi64(precon[2] as i64);
        let w1 = recover_root_si512(wp1, q_v);
        let wp2 = _mm512_set1_epi64(precon[3] as i64);
        let w2 = recover_root_si512(wp2, q_v);

        let mut j = 0usize;
        while j < quarter {
            let p0 = ptr.add(j) as *mut __m512i;
            let p1 = ptr.add(j + quarter) as *mut __m512i;
            let p2 = ptr.add(j + 2 * quarter) as *mut __m512i;
            let p3 = ptr.add(j + 3 * quarter) as *mut __m512i;
            let a = _mm512_loadu_si512(p0 as *const __m512i);
            let b = _mm512_loadu_si512(p1 as *const __m512i);
            let c = _mm512_loadu_si512(p2 as *const __m512i);
            let d = _mm512_loadu_si512(p3 as *const __m512i);

            let tc = harvey_modmul_si512(c, w0, wp0, q_v);
            let td = harvey_modmul_si512(d, w0, wp0, q_v);
            let u0 = _mm512_add_epi64(a, tc);
            let u1 = _mm512_add_epi64(b, td);
            let u2 = _mm512_sub_epi64(_mm512_add_epi64(a, q2_v), tc);
            let u3 = _mm512_sub_epi64(_mm512_add_epi64(b, q2_v), td);

            let t1 = harvey_modmul_si512(u1, w1, wp1, q_v);
            let t3 = harvey_modmul_si512(u3, w2, wp2, q_v);

            _mm512_storeu_si512(p0, _mm512_add_epi64(u0, t1));
            _mm512_storeu_si512(p1, _mm512_sub_epi64(_mm512_add_epi64(u0, q2_v), t1));
            _mm512_storeu_si512(p2, _mm512_add_epi64(u2, t3));
            _mm512_storeu_si512(p3, _mm512_sub_epi64(_mm512_add_epi64(u2, q2_v), t3));
            j += 8;
        }
    }
}

/// Forward NTT (Cooley-Tukey, natural-order input -> bit-reversed output,
/// negacyclic).
///
/// `data` holds 3 contiguous planes of length `n` (`data[k*n .. (k+1)*n]`); each
/// plane is transformed in place with prime `Q[k]` and its scrambled roots.
/// Input is assumed canonical (`[0, q)`). `lazy_output` leaves the result in
/// `[0, 4q)` instead of `[0, q)`, skipping the final reduction; use it for
/// consumers that re-reduce (`c_from_b`, the BBC product, whose bound is `2^44 > 4q`).
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn ntt_avx512<P: PrimeSetNtt3x42Ifma>(table: &Ntt3x42IfmaTable<P>, data: &mut [u64], lazy_output: bool) {
    let n = table.n;
    assert_eq!(data.len(), 3 * n, "data must hold 3 planes of length n");
    if n < 2 {
        return;
    }

    unsafe {
        for k in 0..3 {
            let q = P::Q[k];
            let q2 = 2 * q;
            let plane = &mut data[k * n..(k + 1) * n];
            let root = &table.root[k * n..(k + 1) * n];
            let precon = &table.root_quot[k * n..(k + 1) * n];

            let q_v = _mm512_set1_epi64(q as i64);
            let q2_v = _mm512_set1_epi64(q2 as i64);

            let ptr = plane.as_mut_ptr();

            let tail_p = if n >= 16 {
                let stride = 3 * n / 2;
                &table.tail_quot[k * stride..(k + 1) * stride]
            } else {
                &[]
            };

            if n == 1 << 15 || n == 1 << 16 {
                if n == 1 << 15 {
                    fwd_top2::<{ 1 << 15 }>(ptr, precon, q_v, q2_v);
                } else {
                    fwd_top2::<{ 1 << 16 }>(ptr, precon, q_v, q2_v);
                }
                let quarter = n / 4;
                for part in 0..4 {
                    fwd_plane(
                        ptr.add(part * quarter),
                        quarter,
                        2,
                        part,
                        root,
                        precon,
                        tail_p,
                        q,
                        q2,
                        q_v,
                        q2_v,
                    );
                }
            } else {
                fwd_plane(ptr, n, 0, 0, root, precon, tail_p, q, q2, q_v, q2_v);
            }

            // Final reduction [0, 4q) -> [0, q), skipped on lazy output.
            if !lazy_output {
                let mut off = 0usize;
                while off < n {
                    let xp = ptr.add(off) as *mut __m512i;
                    let x = cond_sub_2q_si512(_mm512_loadu_si512(xp as *const __m512i), q2_v);
                    let x = cond_sub_2q_si512(x, q_v);
                    _mm512_storeu_si512(xp, x);
                    off += 8;
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT (Gentleman-Sande, bit-reversed -> natural)
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse Gentleman-Sande radix-2 butterfly, lazy `[0, 2q)`: given `x`, `y`
/// in `[0, 2q)`, twiddle `w` (`w_precon` its Harvey/Shoup quotient) and
/// `q2 = 2q`, returns `(x', y')` in `[0, 2q)` with `x' = X + Y`,
/// `y' = (X - Y) W (mod q)`.
#[inline]
#[target_feature(enable = "avx512ifma,avx512f")]
unsafe fn inv_butterfly_si512(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    w_precon: __m512i,
    q: __m512i,
    q2: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        let x_out = cond_sub_2q_si512(_mm512_add_epi64(x, y), q2);
        let t = _mm512_sub_epi64(_mm512_add_epi64(x, q2), y); // X+2q-Y in (0,4q)
        let y_out = harvey_modmul_si512(t, w, w_precon, q); // both [0,2q)
        (x_out, y_out)
    }
}

/// Inverse distance-1 interleaved load: gathers the even/odd lanes of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 1` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_inv_interleaved_t1(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let vhi = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
        let vlo = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        let vp2 = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        let a = _mm512_loadu_si512(arg as *const __m512i);
        let b = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let plo = _mm512_permutexvar_epi64(vlo, a);
        let phi = _mm512_permutexvar_epi64(vhi, b);
        let o1 = _mm512_mask_blend_epi64(0x0f, phi, plo);
        let o2 = _mm512_mask_blend_epi64(0xf0, phi, plo);
        let o2 = _mm512_permutexvar_epi64(vp2, o2);
        (o1, o2)
    }
}

/// Inverse distance-2 interleaved load: gathers 2-lane groups of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 2` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_inv_interleaved_t2(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let perm = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
        let a = _mm512_loadu_si512(arg as *const __m512i);
        let b = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let ap = _mm512_permutexvar_epi64(perm, a);
        let bp = _mm512_permutexvar_epi64(perm, b);
        let o1 = _mm512_mask_blend_epi64(0xaa, a, bp);
        let o2 = _mm512_mask_blend_epi64(0xaa, ap, b);
        (o1, o2)
    }
}

/// Inverse distance-4 interleaved load: gathers 4-lane halves of two
/// consecutive registers into the `(x, y)` operand pair for a `t = 4` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_inv_interleaved_t4(arg: *const u64) -> (__m512i, __m512i) {
    unsafe {
        let perm = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        let a = _mm512_loadu_si512(arg as *const __m512i);
        let b = _mm512_loadu_si512(arg.add(8) as *const __m512i);
        let ap = _mm512_permutexvar_epi64(perm, a);
        let bp = _mm512_permutexvar_epi64(perm, b);
        let o1 = _mm512_mask_blend_epi64(0xcc, a, bp);
        let o2 = _mm512_mask_blend_epi64(0xcc, ap, b);
        (o1, o2)
    }
}

/// Inverse permutation of [`load_inv_interleaved_t4`]: scatters the butterfly
/// outputs of a `t = 4` stage back to their natural positions as 4 × `__m256i`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn write_inv_interleaved_t4(x: __m512i, y: __m512i, out: *mut u64) {
    unsafe {
        let x0 = _mm512_extracti64x4_epi64::<0>(x);
        let x1 = _mm512_extracti64x4_epi64::<1>(x);
        let y0 = _mm512_extracti64x4_epi64::<0>(y);
        let y1 = _mm512_extracti64x4_epi64::<1>(y);
        _mm256_storeu_si256(out as *mut __m256i, x0);
        _mm256_storeu_si256(out.add(4) as *mut __m256i, y0);
        _mm256_storeu_si256(out.add(8) as *mut __m256i, x1);
        _mm256_storeu_si256(out.add(12) as *mut __m256i, y1);
    }
}

/// Distance-2 twiddle load: read 4 consecutive u64 roots, broadcast each into
/// 2 lanes to match the operand layout of a `t = 2` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_w_op_t2(arg: *const u64) -> __m512i {
    unsafe {
        let vp = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
        // `vp` only references lanes 0..4, so the upper (undefined) lanes of the
        // cast never leak into the result; no broadcast needed before the permute.
        let w = _mm512_castsi256_si512(_mm256_loadu_si256(arg as *const __m256i));
        _mm512_permutexvar_epi64(vp, w)
    }
}

/// Distance-4 twiddle load: read 2 consecutive u64 roots, broadcast each into
/// 4 lanes to match the operand layout of a `t = 4` stage.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_w_op_t4(arg: *const u64) -> __m512i {
    unsafe {
        let vp = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);
        // `vp` only references lanes 0..2, so the upper (undefined) lanes of the
        // cast never leak into the result. This also avoids `_mm512_broadcast_i64x2`,
        // an AVX-512DQ intrinsic outside this crate's compile-time baseline that
        // would block inlining.
        let w = _mm512_castsi128_si512(_mm_loadu_si128(arg as *const __m128i));
        _mm512_permutexvar_epi64(vp, w)
    }
}

/// Broadcast-twiddle inverse stage: distance `t`, `m` twiddle groups, reading
/// roots from `inv[wi..]`. Each group does `t/8` independent vector butterflies.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_broadcast_stage(ptr: *mut u64, t: usize, m: usize, ip: &[u64], wi: usize, q_v: __m512i, q2_v: __m512i) {
    unsafe {
        for i in 0..m {
            let wp_v = _mm512_set1_epi64(ip[wi + i] as i64);
            let w_v = recover_root_si512(wp_v, q_v);
            let j1 = i * (2 * t);
            let mut j = 0usize;
            while j < t {
                let xp = ptr.add(j1 + j) as *mut __m512i;
                let yp = ptr.add(j1 + j + t) as *mut __m512i;
                let x = _mm512_loadu_si512(xp as *const __m512i);
                let y = _mm512_loadu_si512(yp as *const __m512i);
                let (x, y) = inv_butterfly_si512(x, y, w_v, wp_v, q_v, q2_v);
                _mm512_storeu_si512(xp, x);
                _mm512_storeu_si512(yp, y);
                j += 8;
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_broadcast_pair(ptr: *mut u64, t: usize, m: usize, ip: &[u64], wi: usize, next_wi: usize, q: __m512i, q2: __m512i) {
    unsafe {
        for i in 0..m / 2 {
            let wp0 = _mm512_set1_epi64(ip[wi + 2 * i] as i64);
            let w0 = recover_root_si512(wp0, q);
            let wp1 = _mm512_set1_epi64(ip[wi + 2 * i + 1] as i64);
            let w1 = recover_root_si512(wp1, q);
            let wp2 = _mm512_set1_epi64(ip[next_wi + i] as i64);
            let w2 = recover_root_si512(wp2, q);
            let group = ptr.add(i * 4 * t);
            let mut j = 0;
            while j < t {
                let p0 = group.add(j) as *mut __m512i;
                let p1 = group.add(j + t) as *mut __m512i;
                let p2 = group.add(j + 2 * t) as *mut __m512i;
                let p3 = group.add(j + 3 * t) as *mut __m512i;
                let a = _mm512_loadu_si512(p0);
                let b = _mm512_loadu_si512(p1);
                let c = _mm512_loadu_si512(p2);
                let d = _mm512_loadu_si512(p3);
                // First-stage sums stay below 4q; narrow only their combined sum.
                let q4 = _mm512_add_epi64(q2, q2);
                let ab = _mm512_add_epi64(a, b);
                let cd = _mm512_add_epi64(c, d);
                let b = harvey_modmul_si512(_mm512_sub_epi64(_mm512_add_epi64(a, q2), b), w0, wp0, q);
                let d = harvey_modmul_si512(_mm512_sub_epi64(_mm512_add_epi64(c, q2), d), w1, wp1, q);
                let a = cond_sub_2q_si512(cond_sub_2q_si512(_mm512_add_epi64(ab, cd), q4), q2);
                let c = harvey_modmul_si512(_mm512_sub_epi64(_mm512_add_epi64(ab, q4), cd), w2, wp2, q);
                let (b, d) = inv_butterfly_si512(b, d, w2, wp2, q, q2);
                _mm512_storeu_si512(p0, a);
                _mm512_storeu_si512(p1, b);
                _mm512_storeu_si512(p2, c);
                _mm512_storeu_si512(p3, d);
                j += 8;
            }
        }
    }
}

/// Breadth-first inverse transform of one sub-plane of length `n_sub`.
///
/// Runs all stages except the single `m = 1` (distance `n_sub/2`) stage, which
/// the caller performs (a parent recursion level, or the `1/n`-folding final
/// pass at depth 0). `(depth, half)` locate roots in the shared full-`n` tables.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_plane_base(ptr: *mut u64, n_sub: usize, depth: u32, half: usize, ip: &[u64], q_v: __m512i, q2_v: __m512i) {
    unsafe {
        let mut m = n_sub / 2;
        let mut wi = 1 + m * half;
        // Root-index increment between stages, halved as m halves.
        let mut wi_delta = (m / 2) * ((1usize << (depth + 1)) - half);

        // t = 1 stage, m = n_sub/2. One vectorized root per butterfly.
        {
            let mut j1 = 0usize;
            let mut iter = 0usize;
            while iter < m / 8 {
                let (vx, vy) = load_inv_interleaved_t1(ptr.add(j1));
                let vwp = _mm512_loadu_si512(ip.as_ptr().add(wi + 8 * iter) as *const __m512i);
                let vw = recover_root_si512(vwp, q_v);
                let (vx, vy) = inv_butterfly_si512(vx, vy, vw, vwp, q_v, q2_v);
                _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                j1 += 16;
                iter += 1;
            }
            m >>= 1;
            wi += wi_delta;
            wi_delta >>= 1;
        }

        // t = 2 stage, m = n_sub/4. Each root duplicated into 2 lanes.
        {
            let mut j1 = 0usize;
            let mut iter = 0usize;
            while iter < m / 4 {
                let (vx, vy) = load_inv_interleaved_t2(ptr.add(j1));
                let vwp = load_w_op_t2(ip.as_ptr().add(wi + 4 * iter));
                let vw = recover_root_si512(vwp, q_v);
                let (vx, vy) = inv_butterfly_si512(vx, vy, vw, vwp, q_v, q2_v);
                _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                j1 += 16;
                iter += 1;
            }
            m >>= 1;
            wi += wi_delta;
            wi_delta >>= 1;
        }

        // t = 4 stage, m = n_sub/8. Each root duplicated into 4 lanes.
        {
            let mut j1 = 0usize;
            let mut iter = 0usize;
            while iter < m / 2 {
                let (vx, vy) = load_inv_interleaved_t4(ptr.add(j1));
                let vwp = load_w_op_t4(ip.as_ptr().add(wi + 2 * iter));
                let vw = recover_root_si512(vwp, q_v);
                let (vx, vy) = inv_butterfly_si512(vx, vy, vw, vwp, q_v, q2_v);
                write_inv_interleaved_t4(vx, vy, ptr.add(j1));
                j1 += 16;
                iter += 1;
            }
            m >>= 1;
            wi += wi_delta;
            wi_delta >>= 1;
        }

        // Broadcast-twiddle stages: t = 8, 16, …, n_sub/4 (m = n_sub/16 down to 2).
        let mut t = 8usize;
        while m >= 4 {
            inv_broadcast_pair(ptr, t, m, ip, wi, wi + wi_delta, q_v, q2_v);
            m >>= 2;
            wi += wi_delta + (wi_delta >> 1);
            wi_delta >>= 2;
            t *= 4;
        }
        while m > 1 {
            inv_broadcast_stage(ptr, t, m, ip, wi, q_v, q2_v);
            m >>= 1;
            wi += wi_delta;
            wi_delta >>= 1;
            t *= 2;
        }
    }
}

/// Depth-first inverse transform, splitting into two or four cache-sized children.
/// The final distance-`n_sub/2` stage is deferred to the caller.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_plane(ptr: *mut u64, n_sub: usize, depth: u32, half: usize, ip: &[u64], q_v: __m512i, q2_v: __m512i) {
    unsafe {
        if n_sub <= BASE_NTT_SIZE {
            inv_plane_base(ptr, n_sub, depth, half, ip, q_v, q2_v);
            return;
        }
        let split_depth = if n_sub >= 4 * BASE_NTT_SIZE { 2 } else { 1 };
        let parts = 1 << split_depth;
        let child_n = n_sub / parts;
        for part in 0..parts {
            inv_plane(
                ptr.add(part * child_n),
                child_n,
                depth + split_depth,
                half * parts + part,
                ip,
                q_v,
                q2_v,
            );
        }
        let stride = (1usize << (depth + 1)) - half;
        // At m groups, the root offset is 1 + full_n - m * stride.
        let wi = 1 + (n_sub << depth) - parts * stride;
        if parts == 4 {
            inv_broadcast_pair(ptr, n_sub / 8, 4, ip, wi, wi + 2 * stride, q_v, q2_v);
        } else {
            inv_broadcast_stage(ptr, n_sub / 4, 2, ip, wi, q_v, q2_v);
        }
    }
}

/// Inverse NTT (Gentleman-Sande, bit-reversed input -> natural-order output,
/// negacyclic), with the `1/n` scale folded into the final stage.
///
/// `data` holds 3 contiguous planes of length `n` (`data[k*n .. (k+1)*n]`); each
/// plane is transformed in place with prime `Q[k]` and that plane's reordered
/// inverse roots from `table.inv_root` / `table.inv_quot`. Input must be in
/// `[0, q)`; output is left in `[0, 2q)` (congruent mod q to the natural result).
///
/// For `n >= 32`, the last two stages share a pass with the `1/n` scale:
/// the diff lanes use `W' = (W·n_inv) mod q`, and the sum lanes use `n_inv`.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn intt_avx512<P: PrimeSetNtt3x42Ifma>(table: &Ntt3x42IfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    assert_eq!(data.len(), 3 * n, "data must hold 3 planes of length n");
    if n < 2 {
        return;
    }

    unsafe {
        for k in 0..3 {
            let q = P::Q[k];
            let q2 = 2 * q;
            let inv = &table.inv_root[k * n..(k + 1) * n];
            let ip = &table.inv_quot[k * n..(k + 1) * n];

            let q_v = _mm512_set1_epi64(q as i64);
            let q2_v = _mm512_set1_epi64(q2 as i64);

            let plane = &mut data[k * n..(k + 1) * n];
            let ptr = plane.as_mut_ptr();

            if n < 16 {
                // Scalar fallback for small planes.
                let mut t = 1usize;
                let mut m = n;
                let mut wi = 1usize;
                while m > 1 {
                    let h = m / 2;
                    let mut j1 = 0usize;
                    for _i in 0..h {
                        let w = inv[wi];
                        let w_precon = ip[wi];
                        wi += 1;
                        for j in j1..j1 + t {
                            let x = plane[j];
                            let y = plane[j + t];
                            plane[j] = cond_sub_2q(x + y, q2);
                            plane[j + t] = harvey_modmul(x + q2 - y, w, w_precon, q);
                        }
                        j1 += 2 * t;
                    }
                    t *= 2;
                    m /= 2;
                }
            } else {
                // 1/n scale, folded into the final m == 1 stage below.
                let n_inv = table.n_inv[k];
                let n_inv_quot = table.n_inv_quot[k];
                let n_inv_v = _mm512_set1_epi64(n_inv as i64);
                let n_inv_quot_v = _mm512_set1_epi64(n_inv_quot as i64);

                let w_scaled_v = _mm512_set1_epi64(table.final_root[k] as i64);
                let wp_scaled_v = _mm512_set1_epi64(table.final_quot[k] as i64);
                if n >= 32 {
                    let quarter = n / 4;
                    inv_plane(ptr, n / 2, 1, 0, ip, q_v, q2_v);
                    inv_plane(ptr.add(n / 2), n / 2, 1, 1, ip, q_v, q2_v);
                    let w0 = _mm512_set1_epi64(inv[n - 3] as i64);
                    let wp0 = _mm512_set1_epi64(ip[n - 3] as i64);
                    let w1 = _mm512_set1_epi64(inv[n - 2] as i64);
                    let wp1 = _mm512_set1_epi64(ip[n - 2] as i64);
                    for j in (0..quarter).step_by(8) {
                        let p0 = ptr.add(j) as *mut __m512i;
                        let p1 = ptr.add(j + quarter) as *mut __m512i;
                        let p2 = ptr.add(j + 2 * quarter) as *mut __m512i;
                        let p3 = ptr.add(j + 3 * quarter) as *mut __m512i;
                        let a = _mm512_loadu_si512(p0);
                        let b = _mm512_loadu_si512(p1);
                        let c = _mm512_loadu_si512(p2);
                        let d = _mm512_loadu_si512(p3);
                        let ab = _mm512_add_epi64(a, b);
                        let cd = _mm512_add_epi64(c, d);
                        let b = harvey_modmul_si512(_mm512_sub_epi64(_mm512_add_epi64(a, q2_v), b), w0, wp0, q_v);
                        let d = harvey_modmul_si512(_mm512_sub_epi64(_mm512_add_epi64(c, q2_v), d), w1, wp1, q_v);
                        let a = ab;
                        let c = cd;
                        let ac = harvey_modmul_si512(_mm512_add_epi64(a, c), n_inv_v, n_inv_quot_v, q_v);
                        let bd = harvey_modmul_si512(_mm512_add_epi64(b, d), n_inv_v, n_inv_quot_v, q_v);
                        let ac_diff = _mm512_sub_epi64(_mm512_add_epi64(a, _mm512_add_epi64(q2_v, q2_v)), c);
                        let bd_diff = _mm512_sub_epi64(_mm512_add_epi64(b, q2_v), d);
                        let ac_diff = harvey_modmul_si512(ac_diff, w_scaled_v, wp_scaled_v, q_v);
                        let bd_diff = harvey_modmul_si512(bd_diff, w_scaled_v, wp_scaled_v, q_v);
                        _mm512_storeu_si512(p0, ac);
                        _mm512_storeu_si512(p1, bd);
                        _mm512_storeu_si512(p2, ac_diff);
                        _mm512_storeu_si512(p3, bd_diff);
                    }
                } else {
                    // Depth-first transform of the full plane, down to (but not
                    // including) the single m == 1 stage.
                    inv_plane(ptr, n, 0, 0, ip, q_v, q2_v);

                    // Final stage (m == 1, t = n/2): fold the 1/n scale.
                    let t = n / 2;
                    let mut j = 0usize;
                    while j < t {
                        let xp = ptr.add(j) as *mut __m512i;
                        let yp = ptr.add(j + t) as *mut __m512i;
                        let x = _mm512_loadu_si512(xp as *const __m512i);
                        let y = _mm512_loadu_si512(yp as *const __m512i);
                        // sum lane: (X+Y)·n_inv; diff lane: (X-Y)·W' — folds 1/n.
                        let x_out = harvey_modmul_si512(_mm512_add_epi64(x, y), n_inv_v, n_inv_quot_v, q_v);
                        let t_in = _mm512_sub_epi64(_mm512_add_epi64(x, q2_v), y);
                        let y_out = harvey_modmul_si512(t_in, w_scaled_v, wp_scaled_v, q_v);
                        _mm512_storeu_si512(xp, x_out);
                        _mm512_storeu_si512(yp, y_out);
                        j += 8;
                    }
                }
            }

            if n < 16 {
                // Scalar fallback: separate 1/n scaling pass over the plane.
                let n_inv = table.n_inv[k];
                let n_inv_quot = table.n_inv_quot[k];
                for c in plane.iter_mut() {
                    *c = harvey_modmul(*c, n_inv, n_inv_quot, q);
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt3x42_ifma::tables::harvey_quotient;
    use crate::ntt3x42_ifma::{
        primes::Primes42,
        reference::{
            arithmetic::b_ntt3x42_ifma_from_znx64_ref,
            ntt::{intt3x42_ifma_ref, ntt3x42_ifma_ref},
        },
        tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    };
    use poulpy_hal::layouts::PrimeSet;

    #[test]
    fn harvey_modmul_simd_vs_scalar() {
        use crate::ntt3x42_ifma::tables::{harvey_modmul, harvey_quotient};

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
    fn harvey_modmul_si512_full_range() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..1024 {
            let q: [u64; 8] = std::array::from_fn(|i| Primes42::Q[i % 3]);
            let a: [u64; 8] = std::array::from_fn(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                match case {
                    0 => [0, 1, q[i] - 1, q[i], 2 * q[i] - 1, 4 * q[i] - 1, 8 * q[i] - 1, (1 << 52) - 1][i],
                    _ => state & ((1 << 52) - 1),
                }
            });
            for omega in [0, 1, 7, Primes42::Q[2] / 2, Primes42::Q[2] - 1] {
                let quot: [u64; 8] = std::array::from_fn(|i| harvey_quotient(omega, q[i]));
                let mut got = [0u64; 8];
                unsafe {
                    let result = harvey_modmul_si512(
                        _mm512_loadu_si512(a.as_ptr() as *const __m512i),
                        _mm512_set1_epi64(omega as i64),
                        _mm512_loadu_si512(quot.as_ptr() as *const __m512i),
                        _mm512_loadu_si512(q.as_ptr() as *const __m512i),
                    );
                    _mm512_storeu_si512(got.as_mut_ptr() as *mut __m512i, result);
                }
                for i in 0..8 {
                    assert_eq!(got[i], harvey_modmul(a[i], omega, quot[i], q[i]));
                    assert_eq!(got[i] % q[i], ((a[i] as u128 * omega as u128) % q[i] as u128) as u64);
                    assert!(got[i] < 2 * q[i]);
                }
            }
        }
    }

    #[test]
    fn recover_root_si512_matches_nonzero_twiddles() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for q in Primes42::Q {
            for case in 0..1024 {
                let roots: [u64; 8] = std::array::from_fn(|i| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if case == 0 {
                        [1, 2, 3, q / 2, q / 2 + 1, q - 3, q - 2, q - 1][i]
                    } else {
                        1 + state % (q - 1)
                    }
                });
                let quotients = roots.map(|w| harvey_quotient(w, q));
                let mut actual = [0u64; 8];
                unsafe {
                    let roots = recover_root_si512(_mm512_loadu_si512(quotients.as_ptr().cast()), _mm512_set1_epi64(q as i64));
                    _mm512_storeu_si512(actual.as_mut_ptr().cast(), roots);
                }
                assert_eq!(actual, roots);
            }
        }
    }

    #[test]
    fn ntt_full_range_residues_vs_ref() {
        for log_n in 3..=16 {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
            let inv = Ntt3x42IfmaTableInv::<Primes42>::new(n);
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let input: Vec<u64> = (0..3 * n)
                .map(|i| {
                    let q = Primes42::Q[i / n];
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    if i % 16 < 8 {
                        [0, 1, q - 1, q / 2, q - 2, q / 2 + 1, q / 3, q - 3][i % 8]
                    } else {
                        state % q
                    }
                })
                .collect();
            let mut reference = input.clone();
            ntt3x42_ifma_ref(&fwd, &mut reference);
            for lazy in [false, true] {
                let mut actual = input.clone();
                unsafe {
                    ntt_avx512(&fwd, &mut actual, lazy);
                }
                for i in 0..3 * n {
                    let q = Primes42::Q[i / n];
                    assert_eq!(actual[i] % q, reference[i] % q, "forward n={n} i={i}");
                    assert!(actual[i] < if lazy { 4 * q } else { q });
                    actual[i] %= q;
                }
                let mut expected = actual.clone();
                intt3x42_ifma_ref(&inv, &mut expected);
                unsafe {
                    intt_avx512(&inv, &mut actual);
                }
                for i in 0..3 * n {
                    let q = Primes42::Q[i / n];
                    assert_eq!(actual[i] % q, expected[i] % q, "inverse n={n} i={i}");
                    assert_eq!(actual[i] % q, input[i], "roundtrip n={n} i={i}");
                    assert!(actual[i] < 2 * q);
                }
            }
        }
    }

    #[test]
    fn ntt_uniform_edges_vs_ref() {
        for log_n in 3..=16 {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
            let inv = Ntt3x42IfmaTableInv::<Primes42>::new(n);
            for edge in [0, 1] {
                let input: Vec<u64> = (0..3 * n).map(|i| edge * (Primes42::Q[i / n] - 1)).collect();
                let mut expected = input.clone();
                ntt3x42_ifma_ref(&fwd, &mut expected);
                for lazy in [false, true] {
                    let mut actual = input.clone();
                    unsafe { ntt_avx512(&fwd, &mut actual, lazy) };
                    for i in 0..3 * n {
                        let q = Primes42::Q[i / n];
                        assert_eq!(actual[i] % q, expected[i] % q, "forward n={n} edge={edge} i={i}");
                        assert!(actual[i] < if lazy { 4 * q } else { q });
                    }
                }
                let mut expected = input.clone();
                let mut actual = input;
                intt3x42_ifma_ref(&inv, &mut expected);
                unsafe { intt_avx512(&inv, &mut actual) };
                for i in 0..3 * n {
                    let q = Primes42::Q[i / n];
                    assert_eq!(actual[i] % q, expected[i] % q, "inverse n={n} edge={edge} i={i}");
                    assert!(actual[i] < 2 * q);
                }
            }
        }
    }

    #[test]
    fn ntt_avx512_vs_ref() {
        // NTT3x42Ifma operates on n >= 8 (enforced by Module::new); the kernel is
        // validated against the scalar reference over that supported range.
        for log_n in 3..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 3 * n];
            let mut data_ref = vec![0u64; 3 * n];
            b_ntt3x42_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
            b_ntt3x42_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx512::<Primes42>(&fwd, &mut data_avx, false) };
            ntt3x42_ifma_ref::<Primes42>(&fwd, &mut data_ref);

            // The AVX512 forward reduces fully to [0, q); the reference leaves
            // lazy [0, 2q). Compare residues mod q (per plane).
            for i in 0..3 * n {
                let q = Primes42::Q[i / n];
                assert_eq!(
                    data_avx[i] % q,
                    data_ref[i] % q,
                    "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i],
                    data_ref[i]
                );
            }
        }
    }

    /// Lazy forward output reduced mod q must equal the fully-reduced forward,
    /// and stay within `[0, 4q)`.
    #[test]
    fn ntt_avx512_lazy_output_matches_full() {
        for log_n in [4usize, 8, 11, 13] {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
            let coeffs = pseudorandom_coeffs(n);

            let mut full = vec![0u64; 3 * n];
            let mut lazy = vec![0u64; 3 * n];
            b_ntt3x42_ifma_from_znx64_ref(n, &mut full, &coeffs);
            b_ntt3x42_ifma_from_znx64_ref(n, &mut lazy, &coeffs);

            unsafe {
                ntt_avx512::<Primes42>(&fwd, &mut full, false);
                ntt_avx512::<Primes42>(&fwd, &mut lazy, true);
            }

            for i in 0..3 * n {
                let q = Primes42::Q[i / n];
                assert!(lazy[i] < 4 * q, "n={n} idx={i}: lazy {} not in [0,4q)", lazy[i]);
                assert_eq!(full[i], lazy[i] % q, "n={n} idx={i}: lazy%q != full");
            }
        }
    }

    /// Pseudorandom coefficients in [-10000, 10000) seeded by an LCG.
    fn pseudorandom_coeffs(n: usize) -> Vec<i64> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 11) as i64 % 20001) - 10000
            })
            .collect()
    }

    /// Forward kernel vs scalar reference (mod q) on pseudorandom input.
    /// Sizes > 1024 exercise the depth-first recursion.
    fn ntt_avx512_vs_ref_pseudorandom(n: usize) {
        let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
        let coeffs = pseudorandom_coeffs(n);

        let mut data_avx = vec![0u64; 3 * n];
        let mut data_ref = vec![0u64; 3 * n];
        b_ntt3x42_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
        b_ntt3x42_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

        unsafe { ntt_avx512::<Primes42>(&fwd, &mut data_avx, false) };
        ntt3x42_ifma_ref::<Primes42>(&fwd, &mut data_ref);

        // The AVX512 forward reduces fully to [0, q); the reference leaves
        // lazy [0, 2q). Compare residues mod q (per plane).
        for i in 0..3 * n {
            let q = Primes42::Q[i / n];
            assert_eq!(
                data_avx[i] % q,
                data_ref[i] % q,
                "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                data_avx[i],
                data_ref[i]
            );
        }
    }

    /// Inverse kernel vs scalar reference (mod q) on a forward-transformed
    /// pseudorandom input. Sizes > 1024 exercise the depth-first recursion.
    fn intt_avx512_vs_ref_pseudorandom(n: usize) {
        let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
        let inv = Ntt3x42IfmaTableInv::<Primes42>::new(n);
        let coeffs = pseudorandom_coeffs(n);

        let mut data = vec![0u64; 3 * n];
        b_ntt3x42_ifma_from_znx64_ref(n, &mut data, &coeffs);
        ntt3x42_ifma_ref::<Primes42>(&fwd, &mut data);

        let mut data_avx = data.clone();
        let mut data_ref = data.clone();

        unsafe { intt_avx512::<Primes42>(&inv, &mut data_avx) };
        intt3x42_ifma_ref::<Primes42>(&inv, &mut data_ref);

        for i in 0..3 * n {
            let q = Primes42::Q[i / n];
            assert_eq!(
                data_avx[i] % q,
                data_ref[i] % q,
                "n={n} idx={i}: iNTT AVX512 vs ref (avx={}, ref={})",
                data_avx[i],
                data_ref[i]
            );
        }
    }

    #[test]
    fn ntt_avx512_vs_ref_n4096_pseudorandom() {
        ntt_avx512_vs_ref_pseudorandom(4096);
    }

    #[test]
    fn ntt_avx512_vs_ref_n8192_pseudorandom() {
        ntt_avx512_vs_ref_pseudorandom(8192);
    }

    #[test]
    fn ntt_avx512_vs_ref_n16384_pseudorandom() {
        ntt_avx512_vs_ref_pseudorandom(16384);
    }

    #[test]
    fn ntt_avx512_vs_ref_n32768_pseudorandom() {
        ntt_avx512_vs_ref_pseudorandom(32768);
    }

    #[test]
    fn ntt_avx512_vs_ref_n65536_pseudorandom() {
        ntt_avx512_vs_ref_pseudorandom(65536);
    }

    #[test]
    fn intt_avx512_vs_ref_n4096_pseudorandom() {
        intt_avx512_vs_ref_pseudorandom(4096);
    }

    #[test]
    fn intt_avx512_vs_ref_n8192_pseudorandom() {
        intt_avx512_vs_ref_pseudorandom(8192);
    }

    #[test]
    fn intt_avx512_vs_ref_n16384_pseudorandom() {
        intt_avx512_vs_ref_pseudorandom(16384);
    }

    #[test]
    fn intt_avx512_vs_ref_n32768_pseudorandom() {
        intt_avx512_vs_ref_pseudorandom(32768);
    }

    #[test]
    fn intt_avx512_vs_ref_n65536_pseudorandom() {
        intt_avx512_vs_ref_pseudorandom(65536);
    }

    #[test]
    fn intt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
            let inv = Ntt3x42IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 3 * n];
            b_ntt3x42_ifma_from_znx64_ref(n, &mut data, &coeffs);
            ntt3x42_ifma_ref::<Primes42>(&fwd, &mut data);

            let mut data_avx = data.clone();
            let mut data_ref = data.clone();

            unsafe { intt_avx512::<Primes42>(&inv, &mut data_avx) };
            intt3x42_ifma_ref::<Primes42>(&inv, &mut data_ref);

            // Both kernels leave lazy [0, 2q) output; compare residues mod q.
            for i in 0..3 * n {
                let q = Primes42::Q[i / n];
                assert_eq!(
                    data_avx[i] % q,
                    data_ref[i] % q,
                    "n={n} idx={i}: iNTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i],
                    data_ref[i]
                );
            }
        }
    }

    #[test]
    fn ntt_intt_avx512_roundtrip() {
        // NTT3x42Ifma operates on n >= 8 (enforced by Module::new); forward then
        // inverse recovers the input (mod q) over that supported range.
        for log_n in 3..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt3x42IfmaTable::<Primes42>::new(n);
            let inv = Ntt3x42IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 3 * n];
            b_ntt3x42_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let orig = data.clone();

            unsafe {
                ntt_avx512::<Primes42>(&fwd, &mut data, false);
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
