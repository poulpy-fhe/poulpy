//! Raw AVX512-IFMA forward and inverse NTT kernels, 8 u64 lanes per `__m512i`.
//!
//! Forward: Cooley-Tukey, natural-order input -> bit-reversed output. Inverse:
//! Gentleman-Sande, bit-reversed -> natural, with the `1/n` scale folded in.
//! Butterfly values are kept under a lazy reduction (`[0, 4q)` forward,
//! `[0, 2q)` inverse); the forward kernel ends with a single pass renormalising
//! to `[0, q)`.

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
    _mm512_add_epi64, _mm512_and_si512, _mm512_broadcast_i64x2, _mm512_broadcast_i64x4, _mm512_extracti64x4_epi64,
    _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_blend_epi64, _mm512_min_epu64,
    _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi64,
};

use crate::ntt3x42_ifma::{
    primes::{PrimeSetNtt3x42Ifma, modq_pow64},
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv, cond_sub_2q, harvey_modmul, harvey_quotient},
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
// Forward NTT (Cooley-Tukey, natural -> bit-reversed)
// ──────────────────────────────────────────────────────────────────────────────

/// Forward (decimation-in-time) radix-2 butterfly, lazy `[0, 4q)`: given `x`,
/// `y` in `[0, 4q)`, twiddle `w` (`w_precon` its Harvey/Shoup quotient) and
/// `q2 = 2q`, returns `(x', y')` in `[0, 4q)` with `x' = X + WY`,
/// `y' = X - WY (mod q)`.
#[inline]
#[target_feature(enable = "avx512ifma,avx512f")]
unsafe fn fwd_butterfly_si512(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    w_precon: __m512i,
    q: __m512i,
    q2: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        let x_red = cond_sub_2q_si512(x, q2);
        let t = harvey_modmul_si512(y, w, w_precon, q);
        let x_out = _mm512_add_epi64(x_red, t);
        let y_out = _mm512_add_epi64(x_red, _mm512_sub_epi64(q2, t)); // both [0, 4q)
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
const BASE_NTT_SIZE: usize = 1024;

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
    tail: &[u64],
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
        while t >= 8 {
            let mut j1 = 0usize;
            for i in 0..m {
                let w = root[w_idx + i];
                let w_precon = precon[w_idx + i];
                let w_v = _mm512_set1_epi64(w as i64);
                let w_precon_v = _mm512_set1_epi64(w_precon as i64);

                // t/8 independent vector butterflies over (j, j+t).
                let mut j = j1;
                while j < j1 + t {
                    let xp = ptr.add(j) as *mut __m512i;
                    let yp = ptr.add(j + t) as *mut __m512i;
                    let x_in = _mm512_loadu_si512(xp as *const __m512i);
                    let y_in = _mm512_loadu_si512(yp as *const __m512i);
                    let x_red = cond_sub_2q_si512(x_in, q2_v);
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
                let mut w_ptr = tail.as_ptr().add(off);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t4(ptr.add(j1));
                    let w_v = _mm512_loadu_si512(w_ptr as *const __m512i);
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let (vx, vy) = fwd_butterfly_si512(vx, vy, w_v, wp_v, q_v, q2_v);
                    _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                    _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                    w_ptr = w_ptr.add(8);
                    wp_ptr = wp_ptr.add(8);
                    j1 += 16;
                }
            }

            // t = 2 stage, m = n_sub/4. Distance-2 root block.
            {
                let off = tail_offset(w_idx << 1);
                let mut w_ptr = tail.as_ptr().add(off);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t2(ptr.add(j1));
                    let w_v = _mm512_loadu_si512(w_ptr as *const __m512i);
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let (vx, vy) = fwd_butterfly_si512(vx, vy, w_v, wp_v, q_v, q2_v);
                    _mm512_storeu_si512(ptr.add(j1) as *mut __m512i, vx);
                    _mm512_storeu_si512(ptr.add(j1 + 8) as *mut __m512i, vy);
                    w_ptr = w_ptr.add(8);
                    wp_ptr = wp_ptr.add(8);
                    j1 += 16;
                }
            }

            // t = 1 stage, m = n_sub/2. Distance-1 root block.
            {
                let off = tail_offset(w_idx << 2);
                let mut w_ptr = tail.as_ptr().add(off);
                let mut wp_ptr = tail_p.as_ptr().add(off);
                let mut j1 = 0usize;
                while j1 < n_sub {
                    let (vx, vy) = load_fwd_interleaved_t1(ptr.add(j1));
                    let w_v = _mm512_loadu_si512(w_ptr as *const __m512i);
                    let wp_v = _mm512_loadu_si512(wp_ptr as *const __m512i);
                    let (vx, vy) = fwd_butterfly_si512(vx, vy, w_v, wp_v, q_v, q2_v);
                    write_fwd_interleaved_t1(vx, vy, ptr.add(j1));
                    w_ptr = w_ptr.add(8);
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
/// For `n_sub > BASE_NTT_SIZE`, runs the single top broadcast stage (distance
/// `n_sub/2`, one twiddle group) then recurses into the two halves. For
/// `n_sub <= BASE_NTT_SIZE`, falls through to the breadth-first base case.
///
/// `ILM` (depth-0 only): top-stage input is canonical, so its precondition is skipped.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_plane<const ILM: bool>(
    ptr: *mut u64,
    n_sub: usize,
    depth: u32,
    half: usize,
    root: &[u64],
    precon: &[u64],
    tail: &[u64],
    tail_p: &[u64],
    q: u64,
    q2: u64,
    q_v: __m512i,
    q2_v: __m512i,
) {
    unsafe {
        if n_sub <= BASE_NTT_SIZE {
            // Base case keeps the precondition (small-n top stage isn't worth a split).
            fwd_plane_base(ptr, n_sub, depth, half, root, precon, tail, tail_p, q, q2, q_v, q2_v);
            return;
        }

        // Top broadcast stage: distance t = n_sub/2, single twiddle group.
        let t = n_sub / 2;
        let w_idx = (1usize << depth) + half;
        let w = root[w_idx];
        let w_precon = precon[w_idx];
        let w_v = _mm512_set1_epi64(w as i64);
        let w_precon_v = _mm512_set1_epi64(w_precon as i64);
        let mut j = 0usize;
        while j < t {
            let xp = ptr.add(j) as *mut __m512i;
            let yp = ptr.add(j + t) as *mut __m512i;
            let x_in = _mm512_loadu_si512(xp as *const __m512i);
            let y_in = _mm512_loadu_si512(yp as *const __m512i);
            let x_red = if ILM { x_in } else { cond_sub_2q_si512(x_in, q2_v) };
            let tt = harvey_modmul_si512(y_in, w_v, w_precon_v, q_v);
            let x_out = _mm512_add_epi64(x_red, tt);
            let y_out = _mm512_sub_epi64(_mm512_add_epi64(x_red, q2_v), tt);
            _mm512_storeu_si512(xp, x_out);
            _mm512_storeu_si512(yp, y_out);
            j += 8;
        }

        // Recurse into the two halves (inputs now [0, 4q): no ILM).
        let half_n = n_sub / 2;
        fwd_plane::<false>(ptr, half_n, depth + 1, half * 2, root, precon, tail, tail_p, q, q2, q_v, q2_v);
        fwd_plane::<false>(
            ptr.add(half_n),
            half_n,
            depth + 1,
            half * 2 + 1,
            root,
            precon,
            tail,
            tail_p,
            q,
            q2,
            q_v,
            q2_v,
        );
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
    debug_assert_eq!(data.len(), 3 * n, "data must hold 3 planes of length n");
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

            // Duplicated tail roots for the t = 4, 2, 1 stages (empty for n < 16).
            let (tail, tail_p): (&[u64], &[u64]) = if n >= 16 {
                let stride = 3 * n / 2;
                (
                    &table.tail_root[k * stride..(k + 1) * stride],
                    &table.tail_quot[k * stride..(k + 1) * stride],
                )
            } else {
                (&[], &[])
            };

            // Depth-first transform; the depth-0 top stage skips its precondition.
            fwd_plane::<true>(ptr, n, 0, 0, root, precon, tail, tail_p, q, q2, q_v, q2_v);

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
        let w = _mm512_broadcast_i64x4(_mm256_loadu_si256(arg as *const __m256i));
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
        let w = _mm512_broadcast_i64x2(_mm_loadu_si128(arg as *const __m128i));
        _mm512_permutexvar_epi64(vp, w)
    }
}

/// Broadcast-twiddle inverse stage: distance `t`, `m` twiddle groups, reading
/// roots from `inv[wi..]`. Each group does `t/8` independent vector butterflies.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_broadcast_stage(
    ptr: *mut u64,
    t: usize,
    m: usize,
    inv: &[u64],
    ip: &[u64],
    wi: usize,
    q_v: __m512i,
    q2_v: __m512i,
) {
    unsafe {
        for i in 0..m {
            let w_v = _mm512_set1_epi64(inv[wi + i] as i64);
            let wp_v = _mm512_set1_epi64(ip[wi + i] as i64);
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

/// Breadth-first inverse transform of one sub-plane of length `n_sub`.
///
/// Runs all stages except the single `m = 1` (distance `n_sub/2`) stage, which
/// the caller performs (a parent recursion level, or the `1/n`-folding final
/// pass at depth 0). `(depth, half)` thread the running root index `wi` into the
/// shared full-`n` `inv`/`ip` tables; returns `wi` for the deferred stage.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_plane_base(
    ptr: *mut u64,
    n_sub: usize,
    depth: u32,
    half: usize,
    inv: &[u64],
    ip: &[u64],
    q_v: __m512i,
    q2_v: __m512i,
) -> usize {
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
                let vw = _mm512_loadu_si512(inv.as_ptr().add(wi + 8 * iter) as *const __m512i);
                let vwp = _mm512_loadu_si512(ip.as_ptr().add(wi + 8 * iter) as *const __m512i);
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
                let vw = load_w_op_t2(inv.as_ptr().add(wi + 4 * iter));
                let vwp = load_w_op_t2(ip.as_ptr().add(wi + 4 * iter));
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
                let vw = load_w_op_t4(inv.as_ptr().add(wi + 2 * iter));
                let vwp = load_w_op_t4(ip.as_ptr().add(wi + 2 * iter));
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
        while m > 1 {
            inv_broadcast_stage(ptr, t, m, inv, ip, wi, q_v, q2_v);
            m >>= 1;
            wi += wi_delta;
            wi_delta >>= 1;
            t *= 2;
        }

        // Deferred m = 1 (distance n_sub/2) stage's root index.
        wi
    }
}

/// Depth-first inverse transform of one sub-plane of length `n_sub`.
///
/// For `n_sub > BASE_NTT_SIZE`, recurses into the two halves first, then runs
/// the merge stage (`m = 2`, distance `n_sub/4`); otherwise runs the base case.
/// Either way the single `m = 1` (distance `n_sub/2`) stage is deferred to the
/// caller; its root index is returned.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn inv_plane(
    ptr: *mut u64,
    n_sub: usize,
    depth: u32,
    half: usize,
    inv: &[u64],
    ip: &[u64],
    q_v: __m512i,
    q2_v: __m512i,
) -> usize {
    unsafe {
        if n_sub <= BASE_NTT_SIZE {
            return inv_plane_base(ptr, n_sub, depth, half, inv, ip, q_v, q2_v);
        }

        let half_n = n_sub / 2;
        inv_plane(ptr, half_n, depth + 1, half * 2, inv, ip, q_v, q2_v);
        inv_plane(ptr.add(half_n), half_n, depth + 1, half * 2 + 1, inv, ip, q_v, q2_v);

        // Advance the root index past the stages the halves already ran, then
        // run the single merge stage (m = 2, distance n_sub/4) that finishes
        // both halves' deferred top butterflies.
        let mut m = n_sub / 2;
        let mut wi = 1 + m * half;
        let mut wi_delta = (m / 2) * ((1usize << (depth + 1)) - half);
        let mut t = 1usize;
        while m > 2 {
            t <<= 1;
            wi += wi_delta;
            wi_delta >>= 1;
            m >>= 1;
        }
        // m == 2: merge stage at distance t = n_sub/4.
        inv_broadcast_stage(ptr, t, 2, inv, ip, wi, q_v, q2_v);
        wi += wi_delta;

        // Deferred m = 1 (distance n_sub/2) stage's root index.
        wi
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
/// Each plane runs [`inv_plane`] (depth-first split above `BASE_NTT_SIZE`) down
/// to the single `m = 1` (distance `n/2`) stage, which folds the `1/n` scale:
/// the diff lane uses `W' = (W·n_inv) mod q`, the sum lane is scaled by `n_inv`,
/// avoiding a separate plane sweep.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn intt_avx512<P: PrimeSetNtt3x42Ifma>(table: &Ntt3x42IfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    debug_assert_eq!(data.len(), 3 * n, "data must hold 3 planes of length n");
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
                let n_inv = modq_pow64(n as u64, -1, q);
                let n_inv_quot = harvey_quotient(n_inv, q);
                let n_inv_v = _mm512_set1_epi64(n_inv as i64);
                let n_inv_quot_v = _mm512_set1_epi64(n_inv_quot as i64);

                // Depth-first transform of the full plane, down to (but not
                // including) the single m == 1 stage. Returns that stage's root
                // index.
                let wi = inv_plane(ptr, n, 0, 0, inv, ip, q_v, q2_v);

                // Final stage (m == 1, t = n/2): fold the 1/n scale.
                let t = n / 2;
                let w = inv[wi];
                let w_scaled = ((w as u128 * n_inv as u128) % q as u128) as u64; // W' = W·n_inv
                let w_scaled_v = _mm512_set1_epi64(w_scaled as i64);
                let wp_scaled_v = _mm512_set1_epi64(harvey_quotient(w_scaled, q) as i64);
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

            if n < 16 {
                // Scalar fallback: separate 1/n scaling pass over the plane.
                let n_inv = modq_pow64(n as u64, -1, q);
                let n_inv_quot = harvey_quotient(n_inv, q);
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
    use crate::ntt3x42_ifma::{
        primes::Primes42,
        reference::{
            arithmetic::b_ntt3x42_ifma_from_znx64_ref,
            ntt::{intt3x42_ifma_ref, ntt3x42_ifma_ref},
        },
        tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    };

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
