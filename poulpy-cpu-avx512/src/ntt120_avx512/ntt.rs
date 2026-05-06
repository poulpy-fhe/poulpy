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

//! Q120 NTT forward and inverse — AVX-512F accelerated kernels.
//!
//! Rust port of `q120_ntt_avx2.c` from spqlios-arithmetic, widened from 256-bit to
//! 512-bit by pair-packing two q120b coefficients per `__m512i`.
//!
//! Each q120b coefficient occupies one `__m256i` (4 × u64, one u64 per CRT prime).
//! The 512-bit kernels load two adjacent coefficients per register; per-prime
//! twiddle entries from [`NttTable`] / [`NttTableInv`] (4 × u64 each) are
//! broadcast to both halves with `_mm512_broadcast_i64x4`. A 256-bit tail covers
//! the cases where pair-packing along `i` would only carry a single butterfly.
//!
//! # Algorithm
//!
//! Identical to the scalar reference in [`poulpy_cpu_ref::reference::ntt120::ntt`].
//! The inner loops operate on 4 primes per coefficient (256-bit) and on 2
//! coefficients in parallel (512-bit) where the butterfly shape allows it.
//!
//! Split-precomputed multiplication:
//! ```text
//! split_precompmul(inp, po, h, mask)
//!   = (inp & mask) * (po & 0xFFFF_FFFF)          (low  halves, 32×32→64)
//!   + (inp >> h)   * (po >> 32)                   (high halves, 32×32→64)
//! ```
//! This avoids a full 64×64-bit multiply; the result is congruent to
//! `inp * ω mod Q` up to lazy-reduction multiples of Q.
//!
//! Lazy Barrett reduction:
//! ```text
//! modq_red(x, h, mask, cst)
//!   = (x & mask) + (x >> h) * cst
//! ```
//!
//! # Safety
//!
//! Both public functions require AVX-512F support at runtime (ensured by the
//! [`NTT120Avx512`](super::NTT120Avx512) module constructor).

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64, _mm512_add_epi64,
    _mm512_and_si512, _mm512_broadcast_i64x4, _mm512_loadu_si512, _mm512_mul_epu32, _mm512_permutex2var_epi64, _mm512_set_epi64,
    _mm512_set1_epi64, _mm512_srl_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
};

use poulpy_cpu_ref::reference::ntt120::{
    ntt::{NttReducMeta, NttStepMeta, NttTable, NttTableInv},
    primes::PrimeSet,
};

/// Switch from level-order to block-order processing at this block size.
///
/// Matches `CHANGE_MODE_N` in `q120_ntt_avx2.c`.
const CHANGE_MODE_N: usize = 1024;

// ──────────────────────────────────────────────────────────────────────────────
// Inline 256-bit arithmetic helpers (used by the inner stages where pair-packing
// along `i` is impossible and by tail iterations).
// ──────────────────────────────────────────────────────────────────────────────

/// Split-precomputed multiplication — 4 lanes in one `__m256i`.
///
/// Computes `(inp & mask) * (po & 0xFFFF_FFFF) + (inp >> h) * (po >> 32)`
/// for each of the 4 independent 64-bit lanes.
///
/// `h` is passed as a `__m128i` shift count (lower 64 bits = shift amount)
/// so that `_mm256_srl_epi64` (variable-count form) can be used.
///
/// Matches `split_precompmul_si256` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn split_precompmul_si256(inp: __m256i, po: __m256i, h: __m128i, mask: __m256i) -> __m256i {
    unsafe {
        let inp_low = _mm256_and_si256(inp, mask);
        let t1 = _mm256_mul_epu32(inp_low, po);
        let inp_high = _mm256_srl_epi64(inp, h);
        let po_high = _mm256_srli_epi64(po, 32); // constant shift — compile-time immediate
        let t2 = _mm256_mul_epu32(inp_high, po_high);
        _mm256_add_epi64(t1, t2)
    }
}

/// Lazy Barrett-style modular reduction — 4 lanes in one `__m256i`.
///
/// Computes `(x & mask) + (x >> h) * cst` for each 64-bit lane.
/// The result is congruent to `x mod Q[k]` with reduced bit width.
///
/// Matches `modq_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn modq_red_si256(x: __m256i, h: __m128i, mask: __m256i, cst: __m256i) -> __m256i {
    unsafe {
        let xh = _mm256_srl_epi64(x, h);
        let xl = _mm256_and_si256(x, mask);
        let xh_scaled = _mm256_mul_epu32(xh, cst);
        _mm256_add_epi64(xl, xh_scaled)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 512-bit helpers — pair-pack two q120b coefficients into one __m512i.
// Layout per __m512i: [r0_A, r1_A, r2_A, r3_A,  r0_B, r1_B, r2_B, r3_B].
// Per-prime constants are loaded with `_mm512_set1_epi64` (lane-broadcast) where the
// constant is itself prime-independent, or `_mm512_broadcast_i64x4` from a 4-u64 array
// where it is prime-dependent.
// ──────────────────────────────────────────────────────────────────────────────

#[inline(always)]
unsafe fn split_precompmul_si512(inp: __m512i, po: __m512i, h: __m128i, mask: __m512i) -> __m512i {
    unsafe {
        let inp_low = _mm512_and_si512(inp, mask);
        let t1 = _mm512_mul_epu32(inp_low, po);
        let inp_high = _mm512_srl_epi64(inp, h);
        let po_high = _mm512_srli_epi64::<32>(po);
        let t2 = _mm512_mul_epu32(inp_high, po_high);
        _mm512_add_epi64(t1, t2)
    }
}

#[inline(always)]
unsafe fn modq_red_si512(x: __m512i, h: __m128i, mask: __m512i, cst: __m512i) -> __m512i {
    unsafe {
        let xh = _mm512_srl_epi64(x, h);
        let xl = _mm512_and_si512(x, mask);
        let xh_scaled = _mm512_mul_epu32(xh, cst);
        _mm512_add_epi64(xl, xh_scaled)
    }
}

/// Broadcast the 4 × u64 prime-dependent reduction constant into both halves of a __m512i.
#[inline(always)]
unsafe fn bcast_quad_512(p: *const u64) -> __m512i {
    unsafe { _mm512_broadcast_i64x4(_mm256_loadu_si256(p as *const __m256i)) }
}

/// Cross-block pair-pack indices for `nn = 2` (`halfnn = 1`). Each adjacent pair of blocks
/// `[a_k, b_k, a_{k+1}, b_{k+1}]` is packed via `_mm512_permutex2var_epi64`:
///
/// - `gather_a` picks `[a_k | a_{k+1}]` from `(v0, v1)`.
/// - `gather_b` picks `[b_k | b_{k+1}]` from `(v0, v1)`.
///
/// The same indices, applied to `(out_a_pair, out_b_pair)`, regenerate the original
/// memory layout `[out_a_k, out_b_k]` and `[out_a_{k+1}, out_b_{k+1}]`.
#[inline(always)]
unsafe fn cross_block_idx_a() -> __m512i {
    unsafe { _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0) }
}
#[inline(always)]
unsafe fn cross_block_idx_b() -> __m512i {
    unsafe { _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4) }
}

/// Forward butterfly at `nn = 2` (no twiddle), pair-packed across two adjacent blocks.
///
/// `data` covers a contiguous range of blocks of size 2 (= 8 u64 = one __m512i per block).
/// Returns the byte-pointer advance that was consumed (in `__m256i` units).
#[inline(always)]
unsafe fn ntt_iter_nn2_block_pairs(begin: *mut __m256i, end: *const __m256i, vq2bs_512: __m512i) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        // Each block = 1 __m512i. Two blocks = 2 __m512i = 32 u64 = 8 __m256i.
        let mut data = begin;
        while data.add(4) as usize <= end as usize {
            let v0 = _mm512_loadu_si512(data as *const __m512i); // [a_k, b_k]
            let v1 = _mm512_loadu_si512(data.add(2) as *const __m512i); // [a_{k+1}, b_{k+1}]
            let a_pair = _mm512_permutex2var_epi64(v0, idx_a, v1);
            let b_pair = _mm512_permutex2var_epi64(v0, idx_b, v1);
            let out_a = _mm512_add_epi64(a_pair, b_pair);
            let out_b = _mm512_sub_epi64(_mm512_add_epi64(a_pair, vq2bs_512), b_pair);
            let out_v0 = _mm512_permutex2var_epi64(out_a, idx_a, out_b);
            let out_v1 = _mm512_permutex2var_epi64(out_a, idx_b, out_b);
            _mm512_storeu_si512(data as *mut __m512i, out_v0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, out_v1);
            data = data.add(4);
        }
        data
    }
}

/// Forward butterfly at `nn = 2` with prior reduction.
#[inline(always)]
unsafe fn ntt_iter_red_nn2_block_pairs(
    begin: *mut __m256i,
    end: *const __m256i,
    vq2bs_512: __m512i,
    rh: __m128i,
    rmask_512: __m512i,
    rcst_512: __m512i,
) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        let mut data = begin;
        while data.add(4) as usize <= end as usize {
            let v0 = _mm512_loadu_si512(data as *const __m512i);
            let v1 = _mm512_loadu_si512(data.add(2) as *const __m512i);
            let a_pair = modq_red_si512(_mm512_permutex2var_epi64(v0, idx_a, v1), rh, rmask_512, rcst_512);
            let b_pair = modq_red_si512(_mm512_permutex2var_epi64(v0, idx_b, v1), rh, rmask_512, rcst_512);
            let out_a = _mm512_add_epi64(a_pair, b_pair);
            let out_b = _mm512_sub_epi64(_mm512_add_epi64(a_pair, vq2bs_512), b_pair);
            let out_v0 = _mm512_permutex2var_epi64(out_a, idx_a, out_b);
            let out_v1 = _mm512_permutex2var_epi64(out_a, idx_b, out_b);
            _mm512_storeu_si512(data as *mut __m512i, out_v0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, out_v1);
            data = data.add(4);
        }
        data
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// nn = 4 cross-block pair-pack helpers
//
// Per-block layout: 4 __m256i = `[v0, v1 | v2, v3]` = 2 __m512i words, with
// ptr1 covering `[v0, v1]` (i=0 a, i=1 a) and ptr2 covering `[v2, v3]` (i=0 b,
// i=1 b). Two contiguous blocks A, B form 4 __m512i words. Cross-block pair-
// packing collects the i=0 lanes of both blocks into one __m512i and the i=1
// lanes of both blocks into another, so each butterfly stage runs at full
// 512-bit width on coefficients drawn from two distinct blocks.
//
// idx_a permute selects [lower 256 of arg0 | lower 256 of arg1]; idx_b selects
// the upper halves. The same indices reverse the packing on the way back.
// ──────────────────────────────────────────────────────────────────────────────

/// Forward butterfly at `nn = 4` (one identity + one twiddled butterfly per block),
/// pair-packed across two adjacent blocks. The twiddle ω₁ is identical for every
/// block at this stage and is broadcast to both halves of `vw_512`.
#[inline(always)]
unsafe fn ntt_iter_nn4_block_pairs(
    begin: *mut __m256i,
    end: *const __m256i,
    vq2bs_512: __m512i,
    h: __m128i,
    vmask_512: __m512i,
    vw_512: __m512i,
) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        let mut data = begin;
        while data.add(8) as usize <= end as usize {
            let w0 = _mm512_loadu_si512(data as *const __m512i); // [v0_A, v1_A]
            let w1 = _mm512_loadu_si512(data.add(2) as *const __m512i); // [v2_A, v3_A]
            let w2 = _mm512_loadu_si512(data.add(4) as *const __m512i); // [v0_B, v1_B]
            let w3 = _mm512_loadu_si512(data.add(6) as *const __m512i); // [v2_B, v3_B]

            // Pair-pack i=0 lanes (lower 256 of each ptr1/ptr2 word) and i=1 lanes (upper).
            let a_i0 = _mm512_permutex2var_epi64(w0, idx_a, w2); // [v0_A | v0_B]
            let a_i1 = _mm512_permutex2var_epi64(w0, idx_b, w2); // [v1_A | v1_B]
            let b_i0 = _mm512_permutex2var_epi64(w1, idx_a, w3); // [v2_A | v2_B]
            let b_i1 = _mm512_permutex2var_epi64(w1, idx_b, w3); // [v3_A | v3_B]

            // i = 0: identity butterfly.
            let out_a_i0 = _mm512_add_epi64(a_i0, b_i0);
            let out_b_i0 = _mm512_sub_epi64(_mm512_add_epi64(a_i0, vq2bs_512), b_i0);

            // i = 1: twiddled butterfly.
            let out_a_i1 = _mm512_add_epi64(a_i1, b_i1);
            let b1 = _mm512_sub_epi64(_mm512_add_epi64(a_i1, vq2bs_512), b_i1);
            let out_b_i1 = split_precompmul_si512(b1, vw_512, h, vmask_512);

            // Reverse the pair-pack: same idx_a/idx_b reconstruct memory layout.
            let new_w0 = _mm512_permutex2var_epi64(out_a_i0, idx_a, out_a_i1);
            let new_w2 = _mm512_permutex2var_epi64(out_a_i0, idx_b, out_a_i1);
            let new_w1 = _mm512_permutex2var_epi64(out_b_i0, idx_a, out_b_i1);
            let new_w3 = _mm512_permutex2var_epi64(out_b_i0, idx_b, out_b_i1);

            _mm512_storeu_si512(data as *mut __m512i, new_w0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, new_w1);
            _mm512_storeu_si512(data.add(4) as *mut __m512i, new_w2);
            _mm512_storeu_si512(data.add(6) as *mut __m512i, new_w3);
            data = data.add(8);
        }
        data
    }
}

/// Forward butterfly at `nn = 4` with prior reduction.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ntt_iter_red_nn4_block_pairs(
    begin: *mut __m256i,
    end: *const __m256i,
    vq2bs_512: __m512i,
    h: __m128i,
    vmask_512: __m512i,
    vw_512: __m512i,
    rh: __m128i,
    rmask_512: __m512i,
    rcst_512: __m512i,
) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        let mut data = begin;
        while data.add(8) as usize <= end as usize {
            let w0 = _mm512_loadu_si512(data as *const __m512i);
            let w1 = _mm512_loadu_si512(data.add(2) as *const __m512i);
            let w2 = _mm512_loadu_si512(data.add(4) as *const __m512i);
            let w3 = _mm512_loadu_si512(data.add(6) as *const __m512i);

            let a_i0 = modq_red_si512(_mm512_permutex2var_epi64(w0, idx_a, w2), rh, rmask_512, rcst_512);
            let a_i1 = modq_red_si512(_mm512_permutex2var_epi64(w0, idx_b, w2), rh, rmask_512, rcst_512);
            let b_i0 = modq_red_si512(_mm512_permutex2var_epi64(w1, idx_a, w3), rh, rmask_512, rcst_512);
            let b_i1 = modq_red_si512(_mm512_permutex2var_epi64(w1, idx_b, w3), rh, rmask_512, rcst_512);

            let out_a_i0 = _mm512_add_epi64(a_i0, b_i0);
            let out_b_i0 = _mm512_sub_epi64(_mm512_add_epi64(a_i0, vq2bs_512), b_i0);

            let out_a_i1 = _mm512_add_epi64(a_i1, b_i1);
            let b1 = _mm512_sub_epi64(_mm512_add_epi64(a_i1, vq2bs_512), b_i1);
            let out_b_i1 = split_precompmul_si512(b1, vw_512, h, vmask_512);

            let new_w0 = _mm512_permutex2var_epi64(out_a_i0, idx_a, out_a_i1);
            let new_w2 = _mm512_permutex2var_epi64(out_a_i0, idx_b, out_a_i1);
            let new_w1 = _mm512_permutex2var_epi64(out_b_i0, idx_a, out_b_i1);
            let new_w3 = _mm512_permutex2var_epi64(out_b_i0, idx_b, out_b_i1);

            _mm512_storeu_si512(data as *mut __m512i, new_w0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, new_w1);
            _mm512_storeu_si512(data.add(4) as *mut __m512i, new_w2);
            _mm512_storeu_si512(data.add(6) as *mut __m512i, new_w3);
            data = data.add(8);
        }
        data
    }
}

/// Inverse butterfly at `nn = 4`, pair-packed across two adjacent blocks.
/// Inverse-NTT applies the twiddle to `b` *before* the butterfly:
///   bo = split_precompmul(b, ω₁); (a, bo) → (a + bo, a + q2bs - bo).
#[inline(always)]
unsafe fn intt_iter_nn4_block_pairs(
    begin: *mut __m256i,
    end: *const __m256i,
    vq2bs_512: __m512i,
    h: __m128i,
    vmask_512: __m512i,
    vw_512: __m512i,
) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        let mut data = begin;
        while data.add(8) as usize <= end as usize {
            let w0 = _mm512_loadu_si512(data as *const __m512i);
            let w1 = _mm512_loadu_si512(data.add(2) as *const __m512i);
            let w2 = _mm512_loadu_si512(data.add(4) as *const __m512i);
            let w3 = _mm512_loadu_si512(data.add(6) as *const __m512i);

            let a_i0 = _mm512_permutex2var_epi64(w0, idx_a, w2);
            let a_i1 = _mm512_permutex2var_epi64(w0, idx_b, w2);
            let b_i0 = _mm512_permutex2var_epi64(w1, idx_a, w3);
            let b_i1 = _mm512_permutex2var_epi64(w1, idx_b, w3);

            // i = 0: no twiddle.
            let out_a_i0 = _mm512_add_epi64(a_i0, b_i0);
            let out_b_i0 = _mm512_sub_epi64(_mm512_add_epi64(a_i0, vq2bs_512), b_i0);

            // i = 1: twiddle on b before butterfly.
            let bo = split_precompmul_si512(b_i1, vw_512, h, vmask_512);
            let out_a_i1 = _mm512_add_epi64(a_i1, bo);
            let out_b_i1 = _mm512_sub_epi64(_mm512_add_epi64(a_i1, vq2bs_512), bo);

            let new_w0 = _mm512_permutex2var_epi64(out_a_i0, idx_a, out_a_i1);
            let new_w2 = _mm512_permutex2var_epi64(out_a_i0, idx_b, out_a_i1);
            let new_w1 = _mm512_permutex2var_epi64(out_b_i0, idx_a, out_b_i1);
            let new_w3 = _mm512_permutex2var_epi64(out_b_i0, idx_b, out_b_i1);

            _mm512_storeu_si512(data as *mut __m512i, new_w0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, new_w1);
            _mm512_storeu_si512(data.add(4) as *mut __m512i, new_w2);
            _mm512_storeu_si512(data.add(6) as *mut __m512i, new_w3);
            data = data.add(8);
        }
        data
    }
}

/// Inverse butterfly at `nn = 4` with prior reduction.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn intt_iter_red_nn4_block_pairs(
    begin: *mut __m256i,
    end: *const __m256i,
    vq2bs_512: __m512i,
    h: __m128i,
    vmask_512: __m512i,
    vw_512: __m512i,
    rh: __m128i,
    rmask_512: __m512i,
    rcst_512: __m512i,
) -> *mut __m256i {
    unsafe {
        let idx_a = cross_block_idx_a();
        let idx_b = cross_block_idx_b();
        let mut data = begin;
        while data.add(8) as usize <= end as usize {
            let w0 = _mm512_loadu_si512(data as *const __m512i);
            let w1 = _mm512_loadu_si512(data.add(2) as *const __m512i);
            let w2 = _mm512_loadu_si512(data.add(4) as *const __m512i);
            let w3 = _mm512_loadu_si512(data.add(6) as *const __m512i);

            let a_i0 = modq_red_si512(_mm512_permutex2var_epi64(w0, idx_a, w2), rh, rmask_512, rcst_512);
            let a_i1 = modq_red_si512(_mm512_permutex2var_epi64(w0, idx_b, w2), rh, rmask_512, rcst_512);
            let b_i0 = modq_red_si512(_mm512_permutex2var_epi64(w1, idx_a, w3), rh, rmask_512, rcst_512);
            let b_i1 = modq_red_si512(_mm512_permutex2var_epi64(w1, idx_b, w3), rh, rmask_512, rcst_512);

            let out_a_i0 = _mm512_add_epi64(a_i0, b_i0);
            let out_b_i0 = _mm512_sub_epi64(_mm512_add_epi64(a_i0, vq2bs_512), b_i0);

            let bo = split_precompmul_si512(b_i1, vw_512, h, vmask_512);
            let out_a_i1 = _mm512_add_epi64(a_i1, bo);
            let out_b_i1 = _mm512_sub_epi64(_mm512_add_epi64(a_i1, vq2bs_512), bo);

            let new_w0 = _mm512_permutex2var_epi64(out_a_i0, idx_a, out_a_i1);
            let new_w2 = _mm512_permutex2var_epi64(out_a_i0, idx_b, out_a_i1);
            let new_w1 = _mm512_permutex2var_epi64(out_b_i0, idx_a, out_b_i1);
            let new_w3 = _mm512_permutex2var_epi64(out_b_i0, idx_b, out_b_i1);

            _mm512_storeu_si512(data as *mut __m512i, new_w0);
            _mm512_storeu_si512(data.add(2) as *mut __m512i, new_w1);
            _mm512_storeu_si512(data.add(4) as *mut __m512i, new_w2);
            _mm512_storeu_si512(data.add(6) as *mut __m512i, new_w3);
            data = data.add(8);
        }
        data
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT iteration kernels (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Level-0 forward-NTT pass: multiply each coefficient `a[i]` by `ω^i`.
///
/// Each element of `data` and `powomega` is one q120b coefficient = one `__m256i`.
/// No butterfly — pure element-wise `split_precompmul`. Pair-packs two
/// coefficients per `__m512i`; `count = (end - begin) / sizeof(__m256i)` is
/// guaranteed by the caller to be a power of two ≥ 2, so the loop has no tail.
///
/// Matches `ntt_iter_first` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first(begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, powomega: *const __m256i) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let count = (end as usize - begin as usize) / core::mem::size_of::<__m256i>();
        debug_assert!(count >= 2 && count.is_power_of_two());
        let pairs = count / 2;
        let mut data512 = begin as *mut __m512i;
        let mut po512 = powomega as *const __m512i;
        for _ in 0..pairs {
            let x = _mm512_loadu_si512(data512);
            let po = _mm512_loadu_si512(po512);
            _mm512_storeu_si512(data512, split_precompmul_si512(x, po, h, vmask_512));
            data512 = data512.add(1);
            po512 = po512.add(1);
        }
    }
}

/// Level-0 forward-NTT pass with prior lazy Barrett reduction.
///
/// Like `ntt_iter_first` but each element is reduced via `modq_red` before
/// the `split_precompmul`. Used as the final pass in the inverse NTT when the
/// accumulation bit-width would otherwise exceed 64 bits.
///
/// Matches `ntt_iter_first_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first_red(
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask_512 = _mm512_set1_epi64(reduc.mask as i64);
        let rcst_512 = bcast_quad_512(reduc.modulo_red_cst.as_ptr());
        let count = (end as usize - begin as usize) / core::mem::size_of::<__m256i>();
        debug_assert!(count >= 2 && count.is_power_of_two());
        let pairs = count / 2;
        let mut data512 = begin as *mut __m512i;
        let mut po512 = powomega as *const __m512i;
        let unrolled = pairs / 4;
        for _ in 0..unrolled {
            let x0 = modq_red_si512(_mm512_loadu_si512(data512), rh, rmask_512, rcst_512);
            let x1 = modq_red_si512(_mm512_loadu_si512(data512.add(1)), rh, rmask_512, rcst_512);
            let x2 = modq_red_si512(_mm512_loadu_si512(data512.add(2)), rh, rmask_512, rcst_512);
            let x3 = modq_red_si512(_mm512_loadu_si512(data512.add(3)), rh, rmask_512, rcst_512);
            let po0 = _mm512_loadu_si512(po512);
            let po1 = _mm512_loadu_si512(po512.add(1));
            let po2 = _mm512_loadu_si512(po512.add(2));
            let po3 = _mm512_loadu_si512(po512.add(3));
            _mm512_storeu_si512(data512, split_precompmul_si512(x0, po0, h, vmask_512));
            _mm512_storeu_si512(data512.add(1), split_precompmul_si512(x1, po1, h, vmask_512));
            _mm512_storeu_si512(data512.add(2), split_precompmul_si512(x2, po2, h, vmask_512));
            _mm512_storeu_si512(data512.add(3), split_precompmul_si512(x3, po3, h, vmask_512));
            data512 = data512.add(4);
            po512 = po512.add(4);
        }
        for _ in 0..(pairs & 3) {
            let x = modq_red_si512(_mm512_loadu_si512(data512), rh, rmask_512, rcst_512);
            let po = _mm512_loadu_si512(po512);
            _mm512_storeu_si512(data512, split_precompmul_si512(x, po, h, vmask_512));
            data512 = data512.add(1);
            po512 = po512.add(1);
        }
    }
}

/// Forward Cooley–Tukey (DIT) butterfly level of size `nn`, without reduction.
///
/// For each block of `nn` consecutive coefficients:
/// - `i=0`: `(a, b) → (a+b, a + q2bs - b)` — no twiddle.
/// - `i=1..halfnn-1`: `(a, b) → (a+b, split_precompmul(a + q2bs - b, ω^i))`.
///
/// Matches `ntt_iter` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter(nn: usize, begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, powomega: *const __m256i) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vq2bs_512 = _mm512_broadcast_i64x4(vq2bs);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        // Cross-block pair-pack fast paths for the deepest stages.
        // - nn = 2: 1 butterfly per block, identity twiddle — fully cross-block packed.
        // - nn = 4: 1 identity + 1 twiddled butterfly per block — cross-block packed
        //   with the twiddle (single ω₁ shared across all blocks) broadcast to both halves.
        let mut data = begin;
        if nn == 2 {
            data = ntt_iter_nn2_block_pairs(data, end, vq2bs_512);
        } else if nn == 4 {
            let vw_512 = bcast_quad_512(powomega as *const u64);
            data = ntt_iter_nn4_block_pairs(data, end, vq2bs_512, h, vmask_512, vw_512);
        }
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = _mm256_loadu_si256(ptr1);
            let b = _mm256_loadu_si256(ptr2);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: pair-pack along i when halfnn-1 >= 2 (i.e., halfnn >= 3,
            // so halfnn >= 4 in power-of-2 context). For halfnn = 2, no pairing.
            let mut po_ptr = powomega;
            if halfnn >= 4 {
                let pairs = (halfnn - 2) / 2;
                // 2-way unrolled main loop: two independent split_precompmul chains expose ILP
                // and keep both vector mul ports busy when halfnn ≥ 8.
                let unrolled = pairs / 2;
                for _ in 0..unrolled {
                    let a0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                    let b0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                    let a1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                    let b1v = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                    let po0 = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let po1 = _mm512_loadu_si512(po_ptr.add(2) as *const __m512i);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a0, b0));
                    _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, _mm512_add_epi64(a1, b1v));
                    let d0 = _mm512_sub_epi64(_mm512_add_epi64(a0, vq2bs_512), b0);
                    let d1 = _mm512_sub_epi64(_mm512_add_epi64(a1, vq2bs_512), b1v);
                    _mm512_storeu_si512(ptr2 as *mut __m512i, split_precompmul_si512(d0, po0, h, vmask_512));
                    _mm512_storeu_si512(ptr2.add(2) as *mut __m512i, split_precompmul_si512(d1, po1, h, vmask_512));
                    ptr1 = ptr1.add(4);
                    ptr2 = ptr2.add(4);
                    po_ptr = po_ptr.add(4);
                }
                if pairs & 1 != 0 {
                    let a = _mm512_loadu_si512(ptr1 as *const __m512i);
                    let b = _mm512_loadu_si512(ptr2 as *const __m512i);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a, b));
                    let b1 = _mm512_sub_epi64(_mm512_add_epi64(a, vq2bs_512), b);
                    let po = _mm512_loadu_si512(po_ptr as *const __m512i);
                    _mm512_storeu_si512(ptr2 as *mut __m512i, split_precompmul_si512(b1, po, h, vmask_512));
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                    po_ptr = po_ptr.add(2);
                }
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                let po = _mm256_loadu_si256(po_ptr);
                _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
            } else {
                // halfnn ∈ {1, 2}: 0 or 1 twiddled iterations.
                for _ in 1..halfnn {
                    let a = _mm256_loadu_si256(ptr1);
                    let b = _mm256_loadu_si256(ptr2);
                    _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                    let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                    let po = _mm256_loadu_si256(po_ptr);
                    _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
                    ptr1 = ptr1.add(1);
                    ptr2 = ptr2.add(1);
                    po_ptr = po_ptr.add(1);
                }
            }
            data = data.add(nn);
        }
    }
}

/// Forward Cooley–Tukey butterfly level with prior lazy Barrett reduction.
///
/// Like `ntt_iter` but both `a` and `b` are reduced via `modq_red` before
/// each butterfly, preventing bit-width overflow across levels.
///
/// Matches `ntt_iter_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_red(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vq2bs_512 = _mm512_broadcast_i64x4(vq2bs);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rmask_512 = _mm512_set1_epi64(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);
        let rcst_512 = bcast_quad_512(reduc.modulo_red_cst.as_ptr());

        let mut data = begin;
        // Cross-block pair-pack fast paths for nn ∈ {2, 4} (with prior reduction).
        if nn == 2 {
            data = ntt_iter_red_nn2_block_pairs(data, end, vq2bs_512, rh, rmask_512, rcst_512);
        } else if nn == 4 {
            let vw_512 = bcast_quad_512(powomega as *const u64);
            data = ntt_iter_red_nn4_block_pairs(data, end, vq2bs_512, h, vmask_512, vw_512, rh, rmask_512, rcst_512);
        }
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            let mut po_ptr = powomega;
            if halfnn >= 4 {
                let pairs = (halfnn - 2) / 2;
                let unrolled = pairs / 2;
                for _ in 0..unrolled {
                    let a0 = modq_red_si512(_mm512_loadu_si512(ptr1 as *const __m512i), rh, rmask_512, rcst_512);
                    let b0 = modq_red_si512(_mm512_loadu_si512(ptr2 as *const __m512i), rh, rmask_512, rcst_512);
                    let a1 = modq_red_si512(_mm512_loadu_si512(ptr1.add(2) as *const __m512i), rh, rmask_512, rcst_512);
                    let b1v = modq_red_si512(_mm512_loadu_si512(ptr2.add(2) as *const __m512i), rh, rmask_512, rcst_512);
                    let po0 = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let po1 = _mm512_loadu_si512(po_ptr.add(2) as *const __m512i);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a0, b0));
                    _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, _mm512_add_epi64(a1, b1v));
                    let d0 = _mm512_sub_epi64(_mm512_add_epi64(a0, vq2bs_512), b0);
                    let d1 = _mm512_sub_epi64(_mm512_add_epi64(a1, vq2bs_512), b1v);
                    _mm512_storeu_si512(ptr2 as *mut __m512i, split_precompmul_si512(d0, po0, h, vmask_512));
                    _mm512_storeu_si512(ptr2.add(2) as *mut __m512i, split_precompmul_si512(d1, po1, h, vmask_512));
                    ptr1 = ptr1.add(4);
                    ptr2 = ptr2.add(4);
                    po_ptr = po_ptr.add(4);
                }
                if pairs & 1 != 0 {
                    let a = modq_red_si512(_mm512_loadu_si512(ptr1 as *const __m512i), rh, rmask_512, rcst_512);
                    let b = modq_red_si512(_mm512_loadu_si512(ptr2 as *const __m512i), rh, rmask_512, rcst_512);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a, b));
                    let b1 = _mm512_sub_epi64(_mm512_add_epi64(a, vq2bs_512), b);
                    let po = _mm512_loadu_si512(po_ptr as *const __m512i);
                    _mm512_storeu_si512(ptr2 as *mut __m512i, split_precompmul_si512(b1, po, h, vmask_512));
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                    po_ptr = po_ptr.add(2);
                }
                let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                let po = _mm256_loadu_si256(po_ptr);
                _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
            } else {
                for _ in 1..halfnn {
                    let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                    let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                    _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                    let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                    let po = _mm256_loadu_si256(po_ptr);
                    _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
                    ptr1 = ptr1.add(1);
                    ptr2 = ptr2.add(1);
                    po_ptr = po_ptr.add(1);
                }
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman–Sande (DIF) butterfly level, without reduction.
///
/// For each block of `nn` coefficients:
/// - `i=0`: `(a, b) → (a+b, a + q2bs - b)` — no twiddle.
/// - `i=1..halfnn-1`: twiddle applied to `b` **before** the butterfly:
///   `bo = split_precompmul(b, ω^{-i})`, then `(a, bo) → (a+bo, a + q2bs - bo)`.
///
/// Matches `intt_iter` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn intt_iter(nn: usize, begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, powomega: *const __m256i) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vq2bs_512 = _mm512_broadcast_i64x4(vq2bs);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        // Cross-block pair-pack fast paths for nn ∈ {2, 4} (inverse butterfly).
        let mut data = begin;
        if nn == 2 {
            data = ntt_iter_nn2_block_pairs(data, end, vq2bs_512);
        } else if nn == 4 {
            let vw_512 = bcast_quad_512(powomega as *const u64);
            data = intt_iter_nn4_block_pairs(data, end, vq2bs_512, h, vmask_512, vw_512);
        }
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0
            let a = _mm256_loadu_si256(ptr1);
            let b = _mm256_loadu_si256(ptr2);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            let mut po_ptr = powomega;
            if halfnn >= 4 {
                let pairs = (halfnn - 2) / 2;
                let unrolled = pairs / 2;
                for _ in 0..unrolled {
                    let a0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                    let b0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                    let a1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                    let b1v = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                    let po0 = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let po1 = _mm512_loadu_si512(po_ptr.add(2) as *const __m512i);
                    let bo0 = split_precompmul_si512(b0, po0, h, vmask_512);
                    let bo1 = split_precompmul_si512(b1v, po1, h, vmask_512);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a0, bo0));
                    _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, _mm512_add_epi64(a1, bo1));
                    _mm512_storeu_si512(ptr2 as *mut __m512i, _mm512_sub_epi64(_mm512_add_epi64(a0, vq2bs_512), bo0));
                    _mm512_storeu_si512(
                        ptr2.add(2) as *mut __m512i,
                        _mm512_sub_epi64(_mm512_add_epi64(a1, vq2bs_512), bo1),
                    );
                    ptr1 = ptr1.add(4);
                    ptr2 = ptr2.add(4);
                    po_ptr = po_ptr.add(4);
                }
                if pairs & 1 != 0 {
                    let a = _mm512_loadu_si512(ptr1 as *const __m512i);
                    let b = _mm512_loadu_si512(ptr2 as *const __m512i);
                    let po = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let bo = split_precompmul_si512(b, po, h, vmask_512);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a, bo));
                    _mm512_storeu_si512(ptr2 as *mut __m512i, _mm512_sub_epi64(_mm512_add_epi64(a, vq2bs_512), bo));
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                    po_ptr = po_ptr.add(2);
                }
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let po = _mm256_loadu_si256(po_ptr);
                let bo = split_precompmul_si256(b, po, h, vmask);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
            } else {
                for _ in 1..halfnn {
                    let a = _mm256_loadu_si256(ptr1);
                    let b = _mm256_loadu_si256(ptr2);
                    let po = _mm256_loadu_si256(po_ptr);
                    let bo = split_precompmul_si256(b, po, h, vmask);
                    _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                    _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
                    ptr1 = ptr1.add(1);
                    ptr2 = ptr2.add(1);
                    po_ptr = po_ptr.add(1);
                }
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman–Sande butterfly level with prior lazy Barrett reduction.
///
/// Like `intt_iter` but both `a` and `b` are reduced via `modq_red` before
/// each butterfly.
///
/// Matches `intt_iter_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn intt_iter_red(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vq2bs_512 = _mm512_broadcast_i64x4(vq2bs);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let vmask_512 = _mm512_set1_epi64(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rmask_512 = _mm512_set1_epi64(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);
        let rcst_512 = bcast_quad_512(reduc.modulo_red_cst.as_ptr());

        let mut data = begin;
        // Cross-block pair-pack fast paths for nn ∈ {2, 4} (inverse, with prior reduction).
        if nn == 2 {
            data = ntt_iter_red_nn2_block_pairs(data, end, vq2bs_512, rh, rmask_512, rcst_512);
        } else if nn == 4 {
            let vw_512 = bcast_quad_512(powomega as *const u64);
            data = intt_iter_red_nn4_block_pairs(data, end, vq2bs_512, h, vmask_512, vw_512, rh, rmask_512, rcst_512);
        }
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            let mut po_ptr = powomega;
            if halfnn >= 4 {
                let pairs = (halfnn - 2) / 2;
                let unrolled = pairs / 2;
                for _ in 0..unrolled {
                    let a0 = modq_red_si512(_mm512_loadu_si512(ptr1 as *const __m512i), rh, rmask_512, rcst_512);
                    let b0 = modq_red_si512(_mm512_loadu_si512(ptr2 as *const __m512i), rh, rmask_512, rcst_512);
                    let a1 = modq_red_si512(_mm512_loadu_si512(ptr1.add(2) as *const __m512i), rh, rmask_512, rcst_512);
                    let b1v = modq_red_si512(_mm512_loadu_si512(ptr2.add(2) as *const __m512i), rh, rmask_512, rcst_512);
                    let po0 = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let po1 = _mm512_loadu_si512(po_ptr.add(2) as *const __m512i);
                    let bo0 = split_precompmul_si512(b0, po0, h, vmask_512);
                    let bo1 = split_precompmul_si512(b1v, po1, h, vmask_512);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a0, bo0));
                    _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, _mm512_add_epi64(a1, bo1));
                    _mm512_storeu_si512(ptr2 as *mut __m512i, _mm512_sub_epi64(_mm512_add_epi64(a0, vq2bs_512), bo0));
                    _mm512_storeu_si512(
                        ptr2.add(2) as *mut __m512i,
                        _mm512_sub_epi64(_mm512_add_epi64(a1, vq2bs_512), bo1),
                    );
                    ptr1 = ptr1.add(4);
                    ptr2 = ptr2.add(4);
                    po_ptr = po_ptr.add(4);
                }
                if pairs & 1 != 0 {
                    let a = modq_red_si512(_mm512_loadu_si512(ptr1 as *const __m512i), rh, rmask_512, rcst_512);
                    let b = modq_red_si512(_mm512_loadu_si512(ptr2 as *const __m512i), rh, rmask_512, rcst_512);
                    let po = _mm512_loadu_si512(po_ptr as *const __m512i);
                    let bo = split_precompmul_si512(b, po, h, vmask_512);
                    _mm512_storeu_si512(ptr1 as *mut __m512i, _mm512_add_epi64(a, bo));
                    _mm512_storeu_si512(ptr2 as *mut __m512i, _mm512_sub_epi64(_mm512_add_epi64(a, vq2bs_512), bo));
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                    po_ptr = po_ptr.add(2);
                }
                let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                let po = _mm256_loadu_si256(po_ptr);
                let bo = split_precompmul_si256(b, po, h, vmask);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
            } else {
                for _ in 1..halfnn {
                    let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                    let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                    let po = _mm256_loadu_si256(po_ptr);
                    let bo = split_precompmul_si256(b, po, h, vmask);
                    _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                    _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
                    ptr1 = ptr1.add(1);
                    ptr2 = ptr2.add(1);
                    po_ptr = po_ptr.add(1);
                }
            }
            data = data.add(nn);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward Q120 NTT — AVX-512F accelerated.
///
/// Port of `q120_ntt_bb_avx2` from `q120_ntt_avx2.c`, widened to pair-pack two
/// q120b coefficients per `__m512i`.
///
/// For large transforms (`n > CHANGE_MODE_N`), outer levels are processed
/// sequentially across the full array ("by-level"), then the innermost 1024-wide
/// blocks complete all remaining levels in a single pass ("by-block") to improve
/// cache locality.  For `n ≤ 1024` only the by-block phase runs.
///
/// `data` must be a `u64` slice of length `4 * table.n` in q120b layout.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available (guaranteed by `NTT120Avx512` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn ntt_avx512<P: PrimeSet>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let mut meta_idx = 0usize;
        // po_avx: current offset into powomega, counted in __m256i units
        // (= groups of 4 u64, = one q120b entry).
        let mut po_avx = 0usize;

        // ── Level 0: a[i] *= ω^i (no butterfly) ──────────────────────────
        ntt_iter_first(begin, end, &table.level_metadata[meta_idx], po_base.add(po_avx));
        po_avx += n; // level 0 uses n entries
        meta_idx += 1;

        let split_nn = CHANGE_MODE_N.min(n);

        // ── By-level phase: nn = n, n/2, …, split_nn+1 ───────────────────
        let mut nn = n;
        while nn > split_nn {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                ntt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
            } else {
                ntt_iter(nn, begin, end, meta, po_base.add(po_avx));
            }
            po_avx += halfnn.saturating_sub(1);
            meta_idx += 1;
            nn /= 2;
        }

        // ── By-block phase: process each split_nn-wide block independently ──
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_avx_saved = po_avx;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(split_nn) as *const __m256i;
                meta_idx = meta_idx_saved;
                po_avx = po_avx_saved;
                let mut nn = split_nn;
                while nn >= 2 {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        ntt_iter_red(nn, begin1, end1, meta, po_base.add(po_avx), &table.reduc_metadata);
                    } else {
                        ntt_iter(nn, begin1, end1, meta, po_base.add(po_avx));
                    }
                    po_avx += halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn /= 2;
                }
                it = it.add(split_nn);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse Q120 NTT — AVX-512F accelerated.
///
/// Port of `q120_intt_bb_avx2` from `q120_ntt_avx2.c`, widened to pair-pack two
/// q120b coefficients per `__m512i`.
///
/// The inverse NTT reverses the forward butterfly order (Gentleman–Sande DIF)
/// and finalises with an element-wise multiply by `ω^{-i} * n^{-1}`, which is
/// baked into the last entry of `table.level_metadata` and `table.powomega`.
///
/// Cache-locality strategy mirrors the forward NTT: by-block for the inner
/// `split_nn` levels, by-level for the remaining outer levels.
///
/// `data` must be a `u64` slice of length `4 * table.n` in q120b layout.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available (guaranteed by `NTT120Avx512` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn intt_avx512<P: PrimeSet>(table: &NttTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let mut meta_idx = 0usize;
        let mut po_avx = 0usize;

        let split_nn = CHANGE_MODE_N.min(n);

        // ── By-block phase: levels nn = 2, 4, …, split_nn ────────────────
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_avx_saved = po_avx;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(split_nn) as *const __m256i;
                meta_idx = meta_idx_saved;
                po_avx = po_avx_saved;
                let mut nn = 2usize;
                while nn <= split_nn {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        intt_iter_red(nn, begin1, end1, meta, po_base.add(po_avx), &table.reduc_metadata);
                    } else {
                        intt_iter(nn, begin1, end1, meta, po_base.add(po_avx));
                    }
                    po_avx += halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn *= 2;
                }
                it = it.add(split_nn);
            }
        }

        // ── By-level phase: nn = 2*split_nn, …, n ────────────────────────
        let mut nn = 2 * split_nn;
        while nn <= n {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                intt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
            } else {
                intt_iter(nn, begin, end, meta, po_base.add(po_avx));
            }
            po_avx += halfnn.saturating_sub(1);
            meta_idx += 1;
            nn *= 2;
        }

        // ── Last pass: a[i] *= ω^{-i} * n^{-1} ──────────────────────────
        let meta = &table.level_metadata[meta_idx];
        if meta.reduce {
            ntt_iter_first_red(begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
        } else {
            ntt_iter_first(begin, end, meta, po_base.add(po_avx));
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
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        ntt::{NttTable, NttTableInv, ntt_ref},
        primes::Primes30,
    };

    /// AVX-512F NTT followed by AVX-512F iNTT is the identity — mirrors the ref test.
    #[test]
    fn ntt_intt_identity_avx512() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data, &coeffs);

            let data_orig = data.clone();

            unsafe {
                ntt_avx512::<Primes30>(&fwd, &mut data);
                intt_avx512::<Primes30>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..4 {
                    let orig = data_orig[4 * i + k] % Primes30::Q[k] as u64;
                    let got = data[4 * i + k] % Primes30::Q[k] as u64;
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after AVX-512F NTT+iNTT round-trip");
                }
            }
        }
    }

    /// AVX-512F NTT-based convolution matches known result.
    ///
    /// a = [1, 2, 0, …], b = [3, 4, 0, …]; a*b mod (X^8+1) = [3, 10, 8, 0, …]
    #[test]
    fn ntt_convolution_avx512() {
        let n = 8usize;
        let fwd = NttTable::<Primes30>::new(n);
        let inv = NttTableInv::<Primes30>::new(n);

        let a: Vec<i64> = [1, 2, 0, 0, 0, 0, 0, 0].to_vec();
        let b: Vec<i64> = [3, 4, 0, 0, 0, 0, 0, 0].to_vec();

        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut da, &a);
        b_from_znx64_ref::<Primes30>(n, &mut db, &b);

        unsafe {
            ntt_avx512::<Primes30>(&fwd, &mut da);
            ntt_avx512::<Primes30>(&fwd, &mut db);
        }

        // Pointwise multiply (mod each Q[k])
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..4 {
                let q = Primes30::Q[k] as u64;
                dc[4 * i + k] = (da[4 * i + k] % q * (db[4 * i + k] % q)) % q;
            }
        }

        unsafe {
            intt_avx512::<Primes30>(&inv, &mut dc);
        }

        let mut result = vec![0i128; n];
        b_to_znx128_ref::<Primes30>(n, &mut result, &dc);

        let expected: Vec<i128> = [3, 10, 8, 0, 0, 0, 0, 0].to_vec();
        assert_eq!(result, expected, "AVX-512F NTT convolution mismatch");
    }

    /// AVX-512F NTT output matches reference NTT output.
    #[test]
    fn ntt_avx2_vs_ref() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 13 + 5) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data_avx, &coeffs);
            b_from_znx64_ref::<Primes30>(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx512::<Primes30>(&fwd, &mut data_avx) };
            ntt_ref::<Primes30>(&fwd, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(data_avx[i], data_ref[i], "n={n} idx={i}: NTT AVX-512F vs ref mismatch");
            }
        }
    }
}
