// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! Q120 NTT forward and inverse — AVX2-accelerated kernels.
//!
//! Direct Rust port of `q120_ntt_avx2.c` from spqlios-arithmetic.
//!
//! Each q120b coefficient occupies exactly one `__m256i` (4 × u64, one u64 per
//! CRT prime).  The `powomega` twiddle tables from [`NttTable`] / [`NttTableInv`]
//! share the same 4-u64-per-entry layout and are used directly as `__m256i`
//! pointer arrays.
//!
//! # Algorithm
//!
//! Identical to the scalar reference in [`poulpy_cpu_ref::reference::ntt4x30::ntt`],
//! but the inner loops operate on 4 primes simultaneously via 256-bit SIMD.
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
//! Both public functions require AVX2 support at runtime (ensured by the
//! [`NTT4x30Avx`](super::NTT4x30Avx) module constructor).

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
};

use poulpy_cpu_ref::reference::ntt4x30::{
    ntt::{NttReducMeta, NttStepMeta, NttTable, NttTableInv},
    primes::PrimeSetCrt4,
};

/// Switch from level-order to block-order processing at this block size.
///
/// Matches `CHANGE_MODE_N` in `q120_ntt_avx2.c`.
const CHANGE_MODE_N: usize = 1024;

/// Minimum transform size at which the fused radix-4 by-level kernels are used.
///
/// Radix-4 fusion trades extra register pressure for halving the number of
/// full-array passes in the (memory-bound) by-level phase. It only pays off once
/// the working set (`32·n` bytes) spills out of cache; below this size the simpler
/// radix-2 by-level path has better ILP and is faster, so it is retained. The
/// measured crossover on the dev machine sits between n = 2^14 and n = 2^15 (at
/// n = 2^15/2^16 the fused forward NTT is ~18%/~38% faster, while at n ≤ 2^14 it
/// would regress a few percent). Cache sizes vary, so this is a conservative
/// default rather than a tuned-per-CPU value.
const FUSE_MIN_N: usize = 1 << 15;

// ──────────────────────────────────────────────────────────────────────────────
// Inline AVX2 arithmetic helpers
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
// NTT iteration kernels (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Level-0 forward-NTT pass: multiply each coefficient `a[i]` by `ω^i`.
///
/// Each element of `data` and `powomega` is one q120b coefficient = one `__m256i`.
/// No butterfly — pure element-wise `split_precompmul`.
///
/// Matches `ntt_iter_first` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first(begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, mut powomega: *const __m256i) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = _mm256_loadu_si256(data);
            let po = _mm256_loadu_si256(powomega);
            _mm256_storeu_si256(data, split_precompmul_si256(x, po, h, vmask));
            data = data.add(1);
            powomega = powomega.add(1);
        }
    }
}

/// Level-0 forward-NTT pass with prior lazy Barrett reduction.
///
/// Like `ntt_iter_first` but each element is reduced via `modq_red` before
/// the `split_precompmul`.  Used as the final pass in the inverse NTT when
/// the accumulation bit-width would otherwise exceed 64 bits.
///
/// Matches `ntt_iter_first_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first_red(
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    mut powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = modq_red_si256(_mm256_loadu_si256(data), rh, rmask, rcst);
            let po = _mm256_loadu_si256(powomega);
            _mm256_storeu_si256(data, split_precompmul_si256(x, po, h, vmask));
            data = data.add(1);
            powomega = powomega.add(1);
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
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        let mut data = begin;
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

            // i = 1..halfnn-1: with twiddle on difference
            let mut po_ptr = powomega;
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
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: with twiddle on difference
            let mut po_ptr = powomega;
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
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        let mut data = begin;
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

            // i = 1..halfnn-1: twiddle applied to b before butterfly
            let mut po_ptr = powomega;
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
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: twiddle applied to b before butterfly
            let mut po_ptr = powomega;
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
            data = data.add(nn);
        }
    }
}

/// Conditional lazy Barrett reduction — applies [`modq_red_si256`] iff `red`.
///
/// `red` is loop-invariant in every caller, so the branch is hoisted/predicted;
/// in the memory-bound by-level phase its cost is hidden behind the load/store
/// traffic the fused kernels exist to reduce.
#[inline(always)]
unsafe fn maybe_red_si256(x: __m256i, red: bool, h: __m128i, mask: __m256i, cst: __m256i) -> __m256i {
    if red { unsafe { modq_red_si256(x, h, mask, cst) } } else { x }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fused forward kernels (radix-4 level fusion + level-0 twist folding)
//
// These collapse two adjacent forward butterfly levels into a single load/store
// sweep over the array, halving the (memory-bound) traffic of the by-level phase.
// They are bit-identical to running the equivalent `ntt_iter[_red]` passes back to
// back: the per-coefficient arithmetic and the per-level lazy-reduction schedule
// are unchanged; only the intermediate round-trip through memory is removed.
// ──────────────────────────────────────────────────────────────────────────────

/// Fused forward radix-4 butterfly pass: level `nn` (L1) followed by level `nn/2`
/// (L2), in one sweep. Equivalent to `ntt_iter[_red](nn)` then `ntt_iter[_red](nn/2)`.
///
/// A radix-4 group at offset `j ∈ 0..nn/4` within each `nn`-block touches the four
/// coefficients `{j, j+nn/4, j+nn/2, j+3nn/4}`. L1 butterflies the pairs
/// `(j, j+nn/2)` and `(j+nn/4, j+3nn/4)`; L2 then butterflies `(j, j+nn/4)` (block A)
/// and `(j+nn/2, j+3nn/4)` (block B), both using the same L2 twiddle for offset `j`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ntt_iter_radix4(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta1: &NttStepMeta,
    meta2: &NttStepMeta,
    po1: *const __m256i,
    po2: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let q = nn / 4;
        let vq2bs1 = _mm256_loadu_si256(meta1.q2bs.as_ptr() as *const __m256i);
        let h1 = _mm_cvtsi64_si128(meta1.half_bs as i64);
        let mask1 = _mm256_set1_epi64x(meta1.mask as i64);
        let red1 = meta1.reduce;
        let vq2bs2 = _mm256_loadu_si256(meta2.q2bs.as_ptr() as *const __m256i);
        let h2 = _mm_cvtsi64_si128(meta2.half_bs as i64);
        let mask2 = _mm256_set1_epi64x(meta2.mask as i64);
        let red2 = meta2.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p0 = data;
            let mut p1 = data.add(q);
            let mut p2 = data.add(2 * q);
            let mut p3 = data.add(3 * q);
            let mut t1a = po1; // L1 twiddle for (x0,x2): po1[j-1] (used from j=1)
            let mut t1b = po1.add(q - 1); // L1 twiddle for (x1,x3): po1[j+q-1] (used from j=0)
            let mut t2 = po2; // L2 twiddle for offset j: po2[j-1] (used from j=1)

            // j = 0: (x0,x2) and both L2 pairs have no twiddle; (x1,x3) uses po1[q-1].
            {
                let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red1, rh, rmask, rcst);
                let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red1, rh, rmask, rcst);
                let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red1, rh, rmask, rcst);
                let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red1, rh, rmask, rcst);
                let y0 = _mm256_add_epi64(x0, x2);
                let y2 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs1), x2);
                let y1 = _mm256_add_epi64(x1, x3);
                let d13 = _mm256_sub_epi64(_mm256_add_epi64(x1, vq2bs1), x3);
                let y3 = split_precompmul_si256(d13, _mm256_loadu_si256(t1b), h1, mask1);
                let y0 = maybe_red_si256(y0, red2, rh, rmask, rcst);
                let y1 = maybe_red_si256(y1, red2, rh, rmask, rcst);
                let y2 = maybe_red_si256(y2, red2, rh, rmask, rcst);
                let y3 = maybe_red_si256(y3, red2, rh, rmask, rcst);
                _mm256_storeu_si256(p0, _mm256_add_epi64(y0, y1));
                _mm256_storeu_si256(p1, _mm256_sub_epi64(_mm256_add_epi64(y0, vq2bs2), y1));
                _mm256_storeu_si256(p2, _mm256_add_epi64(y2, y3));
                _mm256_storeu_si256(p3, _mm256_sub_epi64(_mm256_add_epi64(y2, vq2bs2), y3));
                p0 = p0.add(1);
                p1 = p1.add(1);
                p2 = p2.add(1);
                p3 = p3.add(1);
                t1b = t1b.add(1);
            }

            for _ in 1..q {
                let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red1, rh, rmask, rcst);
                let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red1, rh, rmask, rcst);
                let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red1, rh, rmask, rcst);
                let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red1, rh, rmask, rcst);
                let y0 = _mm256_add_epi64(x0, x2);
                let d02 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs1), x2);
                let y2 = split_precompmul_si256(d02, _mm256_loadu_si256(t1a), h1, mask1);
                let y1 = _mm256_add_epi64(x1, x3);
                let d13 = _mm256_sub_epi64(_mm256_add_epi64(x1, vq2bs1), x3);
                let y3 = split_precompmul_si256(d13, _mm256_loadu_si256(t1b), h1, mask1);
                let y0 = maybe_red_si256(y0, red2, rh, rmask, rcst);
                let y1 = maybe_red_si256(y1, red2, rh, rmask, rcst);
                let y2 = maybe_red_si256(y2, red2, rh, rmask, rcst);
                let y3 = maybe_red_si256(y3, red2, rh, rmask, rcst);
                let po2v = _mm256_loadu_si256(t2);
                _mm256_storeu_si256(p0, _mm256_add_epi64(y0, y1));
                let d01 = _mm256_sub_epi64(_mm256_add_epi64(y0, vq2bs2), y1);
                _mm256_storeu_si256(p1, split_precompmul_si256(d01, po2v, h2, mask2));
                _mm256_storeu_si256(p2, _mm256_add_epi64(y2, y3));
                let d23 = _mm256_sub_epi64(_mm256_add_epi64(y2, vq2bs2), y3);
                _mm256_storeu_si256(p3, split_precompmul_si256(d23, po2v, h2, mask2));
                p0 = p0.add(1);
                p1 = p1.add(1);
                p2 = p2.add(1);
                p3 = p3.add(1);
                t1a = t1a.add(1);
                t1b = t1b.add(1);
                t2 = t2.add(1);
            }
            data = data.add(nn);
        }
    }
}

/// Fused forward first pass: level-0 twist (`a[i] *= ω^i`) folded into a radix-4
/// pass over levels `nn=n` (L1) and `n/2` (L2). Equivalent to `ntt_iter_first`
/// then `ntt_iter[_red](n)` then `ntt_iter[_red](n/2)`, in a single sweep.
///
/// The whole array is one `n`-block, so the four coefficients of the radix-4 group
/// at offset `j` sit at absolute positions `{j, j+n/4, j+n/2, j+3n/4}`; the level-0
/// twiddle for each is indexed by its absolute position.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ntt_iter_radix4_first(
    n: usize,
    begin: *mut __m256i,
    meta0: &NttStepMeta,
    meta1: &NttStepMeta,
    meta2: &NttStepMeta,
    po0: *const __m256i,
    po1: *const __m256i,
    po2: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let q = n / 4;
        let h0 = _mm_cvtsi64_si128(meta0.half_bs as i64);
        let mask0 = _mm256_set1_epi64x(meta0.mask as i64);
        let vq2bs1 = _mm256_loadu_si256(meta1.q2bs.as_ptr() as *const __m256i);
        let h1 = _mm_cvtsi64_si128(meta1.half_bs as i64);
        let mask1 = _mm256_set1_epi64x(meta1.mask as i64);
        let red1 = meta1.reduce;
        let vq2bs2 = _mm256_loadu_si256(meta2.q2bs.as_ptr() as *const __m256i);
        let h2 = _mm_cvtsi64_si128(meta2.half_bs as i64);
        let mask2 = _mm256_set1_epi64x(meta2.mask as i64);
        let red2 = meta2.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut p0 = begin;
        let mut p1 = begin.add(q);
        let mut p2 = begin.add(2 * q);
        let mut p3 = begin.add(3 * q);
        // Level-0 twiddles indexed by absolute position of each lane.
        let mut o0 = po0;
        let mut o1 = po0.add(q);
        let mut o2 = po0.add(2 * q);
        let mut o3 = po0.add(3 * q);
        let mut t1a = po1;
        let mut t1b = po1.add(q - 1);
        let mut t2 = po2;

        // j = 0
        {
            let x0 = split_precompmul_si256(_mm256_loadu_si256(p0), _mm256_loadu_si256(o0), h0, mask0);
            let x1 = split_precompmul_si256(_mm256_loadu_si256(p1), _mm256_loadu_si256(o1), h0, mask0);
            let x2 = split_precompmul_si256(_mm256_loadu_si256(p2), _mm256_loadu_si256(o2), h0, mask0);
            let x3 = split_precompmul_si256(_mm256_loadu_si256(p3), _mm256_loadu_si256(o3), h0, mask0);
            let x0 = maybe_red_si256(x0, red1, rh, rmask, rcst);
            let x1 = maybe_red_si256(x1, red1, rh, rmask, rcst);
            let x2 = maybe_red_si256(x2, red1, rh, rmask, rcst);
            let x3 = maybe_red_si256(x3, red1, rh, rmask, rcst);
            let y0 = _mm256_add_epi64(x0, x2);
            let y2 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs1), x2);
            let y1 = _mm256_add_epi64(x1, x3);
            let d13 = _mm256_sub_epi64(_mm256_add_epi64(x1, vq2bs1), x3);
            let y3 = split_precompmul_si256(d13, _mm256_loadu_si256(t1b), h1, mask1);
            let y0 = maybe_red_si256(y0, red2, rh, rmask, rcst);
            let y1 = maybe_red_si256(y1, red2, rh, rmask, rcst);
            let y2 = maybe_red_si256(y2, red2, rh, rmask, rcst);
            let y3 = maybe_red_si256(y3, red2, rh, rmask, rcst);
            _mm256_storeu_si256(p0, _mm256_add_epi64(y0, y1));
            _mm256_storeu_si256(p1, _mm256_sub_epi64(_mm256_add_epi64(y0, vq2bs2), y1));
            _mm256_storeu_si256(p2, _mm256_add_epi64(y2, y3));
            _mm256_storeu_si256(p3, _mm256_sub_epi64(_mm256_add_epi64(y2, vq2bs2), y3));
            p0 = p0.add(1);
            p1 = p1.add(1);
            p2 = p2.add(1);
            p3 = p3.add(1);
            o0 = o0.add(1);
            o1 = o1.add(1);
            o2 = o2.add(1);
            o3 = o3.add(1);
            t1b = t1b.add(1);
        }

        for _ in 1..q {
            let x0 = split_precompmul_si256(_mm256_loadu_si256(p0), _mm256_loadu_si256(o0), h0, mask0);
            let x1 = split_precompmul_si256(_mm256_loadu_si256(p1), _mm256_loadu_si256(o1), h0, mask0);
            let x2 = split_precompmul_si256(_mm256_loadu_si256(p2), _mm256_loadu_si256(o2), h0, mask0);
            let x3 = split_precompmul_si256(_mm256_loadu_si256(p3), _mm256_loadu_si256(o3), h0, mask0);
            let x0 = maybe_red_si256(x0, red1, rh, rmask, rcst);
            let x1 = maybe_red_si256(x1, red1, rh, rmask, rcst);
            let x2 = maybe_red_si256(x2, red1, rh, rmask, rcst);
            let x3 = maybe_red_si256(x3, red1, rh, rmask, rcst);
            let y0 = _mm256_add_epi64(x0, x2);
            let d02 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs1), x2);
            let y2 = split_precompmul_si256(d02, _mm256_loadu_si256(t1a), h1, mask1);
            let y1 = _mm256_add_epi64(x1, x3);
            let d13 = _mm256_sub_epi64(_mm256_add_epi64(x1, vq2bs1), x3);
            let y3 = split_precompmul_si256(d13, _mm256_loadu_si256(t1b), h1, mask1);
            let y0 = maybe_red_si256(y0, red2, rh, rmask, rcst);
            let y1 = maybe_red_si256(y1, red2, rh, rmask, rcst);
            let y2 = maybe_red_si256(y2, red2, rh, rmask, rcst);
            let y3 = maybe_red_si256(y3, red2, rh, rmask, rcst);
            let po2v = _mm256_loadu_si256(t2);
            _mm256_storeu_si256(p0, _mm256_add_epi64(y0, y1));
            let d01 = _mm256_sub_epi64(_mm256_add_epi64(y0, vq2bs2), y1);
            _mm256_storeu_si256(p1, split_precompmul_si256(d01, po2v, h2, mask2));
            _mm256_storeu_si256(p2, _mm256_add_epi64(y2, y3));
            let d23 = _mm256_sub_epi64(_mm256_add_epi64(y2, vq2bs2), y3);
            _mm256_storeu_si256(p3, split_precompmul_si256(d23, po2v, h2, mask2));
            p0 = p0.add(1);
            p1 = p1.add(1);
            p2 = p2.add(1);
            p3 = p3.add(1);
            o0 = o0.add(1);
            o1 = o1.add(1);
            o2 = o2.add(1);
            o3 = o3.add(1);
            t1a = t1a.add(1);
            t1b = t1b.add(1);
            t2 = t2.add(1);
        }
    }
}

/// Fused forward first pass (radix-2): level-0 twist (`a[i] *= ω^i`) folded into
/// the first butterfly level L(n). Equivalent to `ntt_iter_first` then
/// `ntt_iter[_red](n)`, in a single sweep, without the radix-4 register pressure.
///
/// Used for the cache-resident by-level range, where the twist-fold pays off but
/// radix-4 fusion does not. The whole array is one `n`-block, so the level-0
/// twiddle for each coefficient is indexed by its absolute position.
#[inline(always)]
unsafe fn ntt_iter_first_fused(
    n: usize,
    begin: *mut __m256i,
    meta0: &NttStepMeta,
    meta1: &NttStepMeta,
    po0: *const __m256i,
    po1: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfn = n / 2;
        let h0 = _mm_cvtsi64_si128(meta0.half_bs as i64);
        let mask0 = _mm256_set1_epi64x(meta0.mask as i64);
        let vq2bs = _mm256_loadu_si256(meta1.q2bs.as_ptr() as *const __m256i);
        let h1 = _mm_cvtsi64_si128(meta1.half_bs as i64);
        let mask1 = _mm256_set1_epi64x(meta1.mask as i64);
        let red1 = meta1.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut p_a = begin;
        let mut p_b = begin.add(halfn);
        let mut o_a = po0;
        let mut o_b = po0.add(halfn);
        let mut t1 = po1; // L1 twiddle po1[i-1] (used from i=1)

        // i = 0: no L1 twiddle.
        {
            let a = split_precompmul_si256(_mm256_loadu_si256(p_a), _mm256_loadu_si256(o_a), h0, mask0);
            let b = split_precompmul_si256(_mm256_loadu_si256(p_b), _mm256_loadu_si256(o_b), h0, mask0);
            let a = maybe_red_si256(a, red1, rh, rmask, rcst);
            let b = maybe_red_si256(b, red1, rh, rmask, rcst);
            _mm256_storeu_si256(p_a, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(p_b, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            p_a = p_a.add(1);
            p_b = p_b.add(1);
            o_a = o_a.add(1);
            o_b = o_b.add(1);
        }
        for _ in 1..halfn {
            let a = split_precompmul_si256(_mm256_loadu_si256(p_a), _mm256_loadu_si256(o_a), h0, mask0);
            let b = split_precompmul_si256(_mm256_loadu_si256(p_b), _mm256_loadu_si256(o_b), h0, mask0);
            let a = maybe_red_si256(a, red1, rh, rmask, rcst);
            let b = maybe_red_si256(b, red1, rh, rmask, rcst);
            _mm256_storeu_si256(p_a, _mm256_add_epi64(a, b));
            let d = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
            _mm256_storeu_si256(p_b, split_precompmul_si256(d, _mm256_loadu_si256(t1), h1, mask1));
            p_a = p_a.add(1);
            p_b = p_b.add(1);
            o_a = o_a.add(1);
            o_b = o_b.add(1);
            t1 = t1.add(1);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fused inverse kernels (radix-4 level fusion + final normalization folding)
// ──────────────────────────────────────────────────────────────────────────────

/// Fused inverse radix-4 butterfly pass: level `nn` (L_low) followed by level
/// `2*nn` (L_high), in one sweep. Equivalent to `intt_iter[_red](nn)` then
/// `intt_iter[_red](2*nn)`.
///
/// A radix-4 group at offset `j ∈ 0..nn/2` within each `2*nn`-block touches the
/// four coefficients `{j, j+nn/2, j+nn, j+3nn/2}`. L_low (Gentleman–Sande, twiddle
/// applied to `b` first) butterflies `(j, j+nn/2)` (block A) and `(j+nn, j+3nn/2)`
/// (block B) with the same L_low twiddle; L_high then butterflies `(j, j+nn)` and
/// `(j+nn/2, j+3nn/2)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn intt_iter_radix4(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta_lo: &NttStepMeta,
    meta_hi: &NttStepMeta,
    po_lo: *const __m256i,
    po_hi: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let hm = nn / 2;
        let m = nn;
        let blk = 2 * nn;
        let vq2bs_lo = _mm256_loadu_si256(meta_lo.q2bs.as_ptr() as *const __m256i);
        let h_lo = _mm_cvtsi64_si128(meta_lo.half_bs as i64);
        let mask_lo = _mm256_set1_epi64x(meta_lo.mask as i64);
        let red_lo = meta_lo.reduce;
        let vq2bs_hi = _mm256_loadu_si256(meta_hi.q2bs.as_ptr() as *const __m256i);
        let h_hi = _mm_cvtsi64_si128(meta_hi.half_bs as i64);
        let mask_hi = _mm256_set1_epi64x(meta_hi.mask as i64);
        let red_hi = meta_hi.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p0 = data;
            let mut p1 = data.add(hm);
            let mut p2 = data.add(m);
            let mut p3 = data.add(m + hm);
            let mut tlo = po_lo; // L_low twiddle: po_lo[j-1] (used from j=1, both blocks)
            let mut tha = po_hi; // L_high (p0,p2): po_hi[j-1] (used from j=1)
            let mut thb = po_hi.add(hm - 1); // L_high (p1,p3): po_hi[j+hm-1] (used from j=0)

            // j = 0
            {
                let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red_lo, rh, rmask, rcst);
                let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red_lo, rh, rmask, rcst);
                let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red_lo, rh, rmask, rcst);
                let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red_lo, rh, rmask, rcst);
                let z0 = _mm256_add_epi64(x0, x1);
                let z1 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs_lo), x1);
                let z2 = _mm256_add_epi64(x2, x3);
                let z3 = _mm256_sub_epi64(_mm256_add_epi64(x2, vq2bs_lo), x3);
                let z0 = maybe_red_si256(z0, red_hi, rh, rmask, rcst);
                let z1 = maybe_red_si256(z1, red_hi, rh, rmask, rcst);
                let z2 = maybe_red_si256(z2, red_hi, rh, rmask, rcst);
                let z3 = maybe_red_si256(z3, red_hi, rh, rmask, rcst);
                _mm256_storeu_si256(p0, _mm256_add_epi64(z0, z2));
                _mm256_storeu_si256(p2, _mm256_sub_epi64(_mm256_add_epi64(z0, vq2bs_hi), z2));
                let bo = split_precompmul_si256(z3, _mm256_loadu_si256(thb), h_hi, mask_hi);
                _mm256_storeu_si256(p1, _mm256_add_epi64(z1, bo));
                _mm256_storeu_si256(p3, _mm256_sub_epi64(_mm256_add_epi64(z1, vq2bs_hi), bo));
                p0 = p0.add(1);
                p1 = p1.add(1);
                p2 = p2.add(1);
                p3 = p3.add(1);
                thb = thb.add(1);
            }

            for _ in 1..hm {
                let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red_lo, rh, rmask, rcst);
                let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red_lo, rh, rmask, rcst);
                let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red_lo, rh, rmask, rcst);
                let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red_lo, rh, rmask, rcst);
                let tlov = _mm256_loadu_si256(tlo);
                let bo_a = split_precompmul_si256(x1, tlov, h_lo, mask_lo);
                let z0 = _mm256_add_epi64(x0, bo_a);
                let z1 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs_lo), bo_a);
                let bo_b = split_precompmul_si256(x3, tlov, h_lo, mask_lo);
                let z2 = _mm256_add_epi64(x2, bo_b);
                let z3 = _mm256_sub_epi64(_mm256_add_epi64(x2, vq2bs_lo), bo_b);
                let z0 = maybe_red_si256(z0, red_hi, rh, rmask, rcst);
                let z1 = maybe_red_si256(z1, red_hi, rh, rmask, rcst);
                let z2 = maybe_red_si256(z2, red_hi, rh, rmask, rcst);
                let z3 = maybe_red_si256(z3, red_hi, rh, rmask, rcst);
                let bo_p = split_precompmul_si256(z2, _mm256_loadu_si256(tha), h_hi, mask_hi);
                _mm256_storeu_si256(p0, _mm256_add_epi64(z0, bo_p));
                _mm256_storeu_si256(p2, _mm256_sub_epi64(_mm256_add_epi64(z0, vq2bs_hi), bo_p));
                let bo_q = split_precompmul_si256(z3, _mm256_loadu_si256(thb), h_hi, mask_hi);
                _mm256_storeu_si256(p1, _mm256_add_epi64(z1, bo_q));
                _mm256_storeu_si256(p3, _mm256_sub_epi64(_mm256_add_epi64(z1, vq2bs_hi), bo_q));
                p0 = p0.add(1);
                p1 = p1.add(1);
                p2 = p2.add(1);
                p3 = p3.add(1);
                tlo = tlo.add(1);
                tha = tha.add(1);
                thb = thb.add(1);
            }
            data = data.add(blk);
        }
    }
}

/// Fused inverse final pass: radix-4 over the last two butterfly levels `n/2`
/// (L_low) and `n` (L_high), with the final normalization twist (`a[i] *= ω^{-i}/n`)
/// folded into the stores. Equivalent to `intt_iter[_red](n/2)`, `intt_iter[_red](n)`
/// then the element-wise `ntt_iter_first[_red]` last pass, in a single sweep.
///
/// The whole array is one `n`-block, so the four outputs land at absolute positions
/// `{j, j+n/4, j+n/2, j+3n/4}`; the final twiddle for each is indexed by position.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn intt_iter_radix4_last(
    n: usize,
    begin: *mut __m256i,
    meta_lo: &NttStepMeta,
    meta_hi: &NttStepMeta,
    meta_last: &NttStepMeta,
    po_lo: *const __m256i,
    po_hi: *const __m256i,
    po_last: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let hm = n / 4; // = (n/2)/2, half of L_low's block
        let m = n / 2;
        let vq2bs_lo = _mm256_loadu_si256(meta_lo.q2bs.as_ptr() as *const __m256i);
        let h_lo = _mm_cvtsi64_si128(meta_lo.half_bs as i64);
        let mask_lo = _mm256_set1_epi64x(meta_lo.mask as i64);
        let red_lo = meta_lo.reduce;
        let vq2bs_hi = _mm256_loadu_si256(meta_hi.q2bs.as_ptr() as *const __m256i);
        let h_hi = _mm_cvtsi64_si128(meta_hi.half_bs as i64);
        let mask_hi = _mm256_set1_epi64x(meta_hi.mask as i64);
        let red_hi = meta_hi.reduce;
        let h_last = _mm_cvtsi64_si128(meta_last.half_bs as i64);
        let mask_last = _mm256_set1_epi64x(meta_last.mask as i64);
        let red_last = meta_last.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut p0 = begin;
        let mut p1 = begin.add(hm);
        let mut p2 = begin.add(m);
        let mut p3 = begin.add(m + hm);
        let mut tlo = po_lo;
        let mut tha = po_hi;
        let mut thb = po_hi.add(hm - 1);
        // Final-twist twiddles indexed by absolute output position.
        let mut l0 = po_last;
        let mut l1 = po_last.add(hm);
        let mut l2 = po_last.add(m);
        let mut l3 = po_last.add(m + hm);

        // j = 0
        {
            let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red_lo, rh, rmask, rcst);
            let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red_lo, rh, rmask, rcst);
            let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red_lo, rh, rmask, rcst);
            let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red_lo, rh, rmask, rcst);
            let z0 = _mm256_add_epi64(x0, x1);
            let z1 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs_lo), x1);
            let z2 = _mm256_add_epi64(x2, x3);
            let z3 = _mm256_sub_epi64(_mm256_add_epi64(x2, vq2bs_lo), x3);
            let z0 = maybe_red_si256(z0, red_hi, rh, rmask, rcst);
            let z1 = maybe_red_si256(z1, red_hi, rh, rmask, rcst);
            let z2 = maybe_red_si256(z2, red_hi, rh, rmask, rcst);
            let z3 = maybe_red_si256(z3, red_hi, rh, rmask, rcst);
            let o0 = _mm256_add_epi64(z0, z2);
            let o2 = _mm256_sub_epi64(_mm256_add_epi64(z0, vq2bs_hi), z2);
            let bo = split_precompmul_si256(z3, _mm256_loadu_si256(thb), h_hi, mask_hi);
            let o1 = _mm256_add_epi64(z1, bo);
            let o3 = _mm256_sub_epi64(_mm256_add_epi64(z1, vq2bs_hi), bo);
            let o0 = maybe_red_si256(o0, red_last, rh, rmask, rcst);
            let o1 = maybe_red_si256(o1, red_last, rh, rmask, rcst);
            let o2 = maybe_red_si256(o2, red_last, rh, rmask, rcst);
            let o3 = maybe_red_si256(o3, red_last, rh, rmask, rcst);
            _mm256_storeu_si256(p0, split_precompmul_si256(o0, _mm256_loadu_si256(l0), h_last, mask_last));
            _mm256_storeu_si256(p1, split_precompmul_si256(o1, _mm256_loadu_si256(l1), h_last, mask_last));
            _mm256_storeu_si256(p2, split_precompmul_si256(o2, _mm256_loadu_si256(l2), h_last, mask_last));
            _mm256_storeu_si256(p3, split_precompmul_si256(o3, _mm256_loadu_si256(l3), h_last, mask_last));
            p0 = p0.add(1);
            p1 = p1.add(1);
            p2 = p2.add(1);
            p3 = p3.add(1);
            thb = thb.add(1);
            l0 = l0.add(1);
            l1 = l1.add(1);
            l2 = l2.add(1);
            l3 = l3.add(1);
        }

        for _ in 1..hm {
            let x0 = maybe_red_si256(_mm256_loadu_si256(p0), red_lo, rh, rmask, rcst);
            let x1 = maybe_red_si256(_mm256_loadu_si256(p1), red_lo, rh, rmask, rcst);
            let x2 = maybe_red_si256(_mm256_loadu_si256(p2), red_lo, rh, rmask, rcst);
            let x3 = maybe_red_si256(_mm256_loadu_si256(p3), red_lo, rh, rmask, rcst);
            let tlov = _mm256_loadu_si256(tlo);
            let bo_a = split_precompmul_si256(x1, tlov, h_lo, mask_lo);
            let z0 = _mm256_add_epi64(x0, bo_a);
            let z1 = _mm256_sub_epi64(_mm256_add_epi64(x0, vq2bs_lo), bo_a);
            let bo_b = split_precompmul_si256(x3, tlov, h_lo, mask_lo);
            let z2 = _mm256_add_epi64(x2, bo_b);
            let z3 = _mm256_sub_epi64(_mm256_add_epi64(x2, vq2bs_lo), bo_b);
            let z0 = maybe_red_si256(z0, red_hi, rh, rmask, rcst);
            let z1 = maybe_red_si256(z1, red_hi, rh, rmask, rcst);
            let z2 = maybe_red_si256(z2, red_hi, rh, rmask, rcst);
            let z3 = maybe_red_si256(z3, red_hi, rh, rmask, rcst);
            let bo_p = split_precompmul_si256(z2, _mm256_loadu_si256(tha), h_hi, mask_hi);
            let o0 = _mm256_add_epi64(z0, bo_p);
            let o2 = _mm256_sub_epi64(_mm256_add_epi64(z0, vq2bs_hi), bo_p);
            let bo_q = split_precompmul_si256(z3, _mm256_loadu_si256(thb), h_hi, mask_hi);
            let o1 = _mm256_add_epi64(z1, bo_q);
            let o3 = _mm256_sub_epi64(_mm256_add_epi64(z1, vq2bs_hi), bo_q);
            let o0 = maybe_red_si256(o0, red_last, rh, rmask, rcst);
            let o1 = maybe_red_si256(o1, red_last, rh, rmask, rcst);
            let o2 = maybe_red_si256(o2, red_last, rh, rmask, rcst);
            let o3 = maybe_red_si256(o3, red_last, rh, rmask, rcst);
            _mm256_storeu_si256(p0, split_precompmul_si256(o0, _mm256_loadu_si256(l0), h_last, mask_last));
            _mm256_storeu_si256(p1, split_precompmul_si256(o1, _mm256_loadu_si256(l1), h_last, mask_last));
            _mm256_storeu_si256(p2, split_precompmul_si256(o2, _mm256_loadu_si256(l2), h_last, mask_last));
            _mm256_storeu_si256(p3, split_precompmul_si256(o3, _mm256_loadu_si256(l3), h_last, mask_last));
            p0 = p0.add(1);
            p1 = p1.add(1);
            p2 = p2.add(1);
            p3 = p3.add(1);
            tlo = tlo.add(1);
            tha = tha.add(1);
            thb = thb.add(1);
            l0 = l0.add(1);
            l1 = l1.add(1);
            l2 = l2.add(1);
            l3 = l3.add(1);
        }
    }
}

/// Fused inverse final pass (radix-2): last butterfly level L(n) with the final
/// normalization twist (`a[i] *= ω^{-i}/n`) folded into the stores. Equivalent to
/// `intt_iter[_red](n)` then the element-wise `ntt_iter_first[_red]` last pass, in
/// a single sweep, without the radix-4 register pressure.
///
/// Used for the cache-resident by-level range. The whole array is one `n`-block,
/// so the final twiddle for each output is indexed by its absolute position.
#[inline(always)]
unsafe fn intt_iter_last_fused(
    n: usize,
    begin: *mut __m256i,
    meta_bf: &NttStepMeta,
    meta_last: &NttStepMeta,
    po_bf: *const __m256i,
    po_last: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfn = n / 2;
        let vq2bs = _mm256_loadu_si256(meta_bf.q2bs.as_ptr() as *const __m256i);
        let h_bf = _mm_cvtsi64_si128(meta_bf.half_bs as i64);
        let mask_bf = _mm256_set1_epi64x(meta_bf.mask as i64);
        let red_bf = meta_bf.reduce;
        let h_last = _mm_cvtsi64_si128(meta_last.half_bs as i64);
        let mask_last = _mm256_set1_epi64x(meta_last.mask as i64);
        let red_last = meta_last.reduce;
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut p_a = begin;
        let mut p_b = begin.add(halfn);
        let mut l_a = po_last;
        let mut l_b = po_last.add(halfn);
        let mut t = po_bf; // butterfly twiddle po_bf[i-1] (used from i=1)

        // i = 0: no butterfly twiddle.
        {
            let a = maybe_red_si256(_mm256_loadu_si256(p_a), red_bf, rh, rmask, rcst);
            let b = maybe_red_si256(_mm256_loadu_si256(p_b), red_bf, rh, rmask, rcst);
            let oa = _mm256_add_epi64(a, b);
            let ob = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
            let oa = maybe_red_si256(oa, red_last, rh, rmask, rcst);
            let ob = maybe_red_si256(ob, red_last, rh, rmask, rcst);
            _mm256_storeu_si256(p_a, split_precompmul_si256(oa, _mm256_loadu_si256(l_a), h_last, mask_last));
            _mm256_storeu_si256(p_b, split_precompmul_si256(ob, _mm256_loadu_si256(l_b), h_last, mask_last));
            p_a = p_a.add(1);
            p_b = p_b.add(1);
            l_a = l_a.add(1);
            l_b = l_b.add(1);
        }
        for _ in 1..halfn {
            let a = maybe_red_si256(_mm256_loadu_si256(p_a), red_bf, rh, rmask, rcst);
            let b = maybe_red_si256(_mm256_loadu_si256(p_b), red_bf, rh, rmask, rcst);
            let bo = split_precompmul_si256(b, _mm256_loadu_si256(t), h_bf, mask_bf);
            let oa = _mm256_add_epi64(a, bo);
            let ob = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo);
            let oa = maybe_red_si256(oa, red_last, rh, rmask, rcst);
            let ob = maybe_red_si256(ob, red_last, rh, rmask, rcst);
            _mm256_storeu_si256(p_a, split_precompmul_si256(oa, _mm256_loadu_si256(l_a), h_last, mask_last));
            _mm256_storeu_si256(p_b, split_precompmul_si256(ob, _mm256_loadu_si256(l_b), h_last, mask_last));
            p_a = p_a.add(1);
            p_b = p_b.add(1);
            l_a = l_a.add(1);
            l_b = l_b.add(1);
            t = t.add(1);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward Q120 NTT — AVX2 accelerated.
///
/// Direct port of `q120_ntt_bb_avx2` from `q120_ntt_avx2.c`.
///
/// For large transforms (`n > CHANGE_MODE_N = 1024`), outer levels are processed
/// sequentially across the full array ("by-level"), then the innermost 1024-wide
/// blocks complete all remaining levels in a single pass ("by-block") to improve
/// cache locality.  For `n ≤ 1024` only the by-block phase runs.
///
/// `data` must be a `u64` slice of length `4 * table.n` in q120b layout.
///
/// # Safety
///
/// Caller must ensure AVX2 is available (guaranteed by `NTT4x30Avx` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ntt_avx2<P: PrimeSetCrt4>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        // meta_idx / po_avx track the current level's metadata index and the
        // offset into `powomega` (in __m256i units = one q120b entry); both are
        // left pointing at the first by-block level for the shared phase below.
        let mut meta_idx: usize;
        let mut po_avx: usize;

        let split_nn = CHANGE_MODE_N.min(n);

        // ── By-level phase (nn = n, n/2, …, 2*split_nn): fused when possible ──
        if n >= FUSE_MIN_N {
            if n / 2 > split_nn {
                // ≥2 by-level levels: fold the level-0 twist into a radix-4 pass
                // over L(n) and L(n/2), then radix-4 the remaining pairs.
                ntt_iter_radix4_first(
                    n,
                    begin,
                    &table.level_metadata[0],
                    &table.level_metadata[1],
                    &table.level_metadata[2],
                    po_base,
                    po_base.add(n),
                    po_base.add(n + (n / 2 - 1)),
                    &table.reduc_metadata,
                );
                meta_idx = 3;
                po_avx = n + (n / 2 - 1) + (n / 4 - 1);
                let mut nn = n / 4;
                while nn > split_nn {
                    if nn / 2 > split_nn {
                        ntt_iter_radix4(
                            nn,
                            begin,
                            end,
                            &table.level_metadata[meta_idx],
                            &table.level_metadata[meta_idx + 1],
                            po_base.add(po_avx),
                            po_base.add(po_avx + (nn / 2 - 1)),
                            &table.reduc_metadata,
                        );
                        po_avx += (nn / 2 - 1) + (nn / 4 - 1);
                        meta_idx += 2;
                        nn /= 4;
                    } else {
                        // Odd tail level: single radix-2 butterfly.
                        let meta = &table.level_metadata[meta_idx];
                        if meta.reduce {
                            ntt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
                        } else {
                            ntt_iter(nn, begin, end, meta, po_base.add(po_avx));
                        }
                        po_avx += nn / 2 - 1;
                        meta_idx += 1;
                        nn /= 2;
                    }
                }
            } else {
                // Exactly one by-level level (n = 2*split_nn): level-0 twist then a
                // single radix-2 butterfly (no pair to fuse with).
                ntt_iter_first(begin, end, &table.level_metadata[0], po_base);
                let meta = &table.level_metadata[1];
                if meta.reduce {
                    ntt_iter_red(n, begin, end, meta, po_base.add(n), &table.reduc_metadata);
                } else {
                    ntt_iter(n, begin, end, meta, po_base.add(n));
                }
                meta_idx = 2;
                po_avx = n + (n / 2 - 1);
            }
        } else if n > split_nn {
            // Cache-resident by-level phase: radix-2 levels, with the level-0 twist
            // folded into the first butterfly L(n) (radix-4 fusion would regress here,
            // but the twist-fold adds almost no register pressure).
            ntt_iter_first_fused(
                n,
                begin,
                &table.level_metadata[0],
                &table.level_metadata[1],
                po_base,
                po_base.add(n),
                &table.reduc_metadata,
            );
            meta_idx = 2;
            po_avx = n + (n / 2 - 1);
            let mut nn = n / 2;
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
        } else {
            // n ≤ split_nn: no by-level phase; level-0 twist then by-block.
            ntt_iter_first(begin, end, &table.level_metadata[0], po_base);
            meta_idx = 1;
            po_avx = n;
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

/// Inverse Q120 NTT — AVX2 accelerated.
///
/// Direct port of `q120_intt_bb_avx2` from `q120_ntt_avx2.c`.
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
/// Caller must ensure AVX2 is available (guaranteed by `NTT4x30Avx` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn intt_avx2<P: PrimeSetCrt4>(table: &NttTableInv<P>, data: &mut [u64]) {
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

        // ── By-level phase (nn = 2*split_nn, …, n) + final twist: fused ──
        if n >= FUSE_MIN_N {
            // Number of by-level butterfly levels (split_nn == 1024 here).
            let c = (n.trailing_zeros() - split_nn.trailing_zeros()) as usize;
            if c == 1 {
                // Single by-level level (n = 2*split_nn): butterfly L(n) then the
                // element-wise final twist, unfused (no pair to combine with).
                let meta = &table.level_metadata[meta_idx];
                if meta.reduce {
                    intt_iter_red(n, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
                } else {
                    intt_iter(n, begin, end, meta, po_base.add(po_avx));
                }
                po_avx += n / 2 - 1;
                meta_idx += 1;
                let meta = &table.level_metadata[meta_idx];
                if meta.reduce {
                    ntt_iter_first_red(begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
                } else {
                    ntt_iter_first(begin, end, meta, po_base.add(po_avx));
                }
            } else {
                // c ≥ 2: radix-4 the bottom (c-2) levels (single for an odd tail),
                // then fuse the top two levels L(n/2), L(n) with the final twist.
                let mut nn = 2 * split_nn;
                let mut remaining = c - 2;
                while remaining >= 2 {
                    intt_iter_radix4(
                        nn,
                        begin,
                        end,
                        &table.level_metadata[meta_idx],
                        &table.level_metadata[meta_idx + 1],
                        po_base.add(po_avx),
                        po_base.add(po_avx + (nn / 2 - 1)),
                        &table.reduc_metadata,
                    );
                    po_avx += (nn / 2 - 1) + (nn - 1);
                    meta_idx += 2;
                    nn *= 4;
                    remaining -= 2;
                }
                if remaining == 1 {
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        intt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
                    } else {
                        intt_iter(nn, begin, end, meta, po_base.add(po_avx));
                    }
                    po_avx += nn / 2 - 1;
                    meta_idx += 1;
                }
                // Fuse L(n/2), L(n) and the final twist in one sweep.
                intt_iter_radix4_last(
                    n,
                    begin,
                    &table.level_metadata[meta_idx],
                    &table.level_metadata[meta_idx + 1],
                    &table.level_metadata[meta_idx + 2],
                    po_base.add(po_avx),
                    po_base.add(po_avx + (n / 4 - 1)),
                    po_base.add(po_avx + (n / 4 - 1) + (n / 2 - 1)),
                    &table.reduc_metadata,
                );
            }
        } else if n > split_nn {
            // Cache-resident by-level phase: radix-2 levels, with the final
            // normalization twist folded into the last butterfly L(n).
            let mut nn = 2 * split_nn;
            while nn < n {
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
            intt_iter_last_fused(
                n,
                begin,
                &table.level_metadata[meta_idx],
                &table.level_metadata[meta_idx + 1],
                po_base.add(po_avx),
                po_base.add(po_avx + (n / 2 - 1)),
                &table.reduc_metadata,
            );
        } else {
            // n ≤ split_nn: by-block already ran every butterfly level; final twist.
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                ntt_iter_first_red(begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
            } else {
                ntt_iter_first(begin, end, meta, po_base.add(po_avx));
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt4x30::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        ntt::{NttTable, NttTableInv, intt_ref, ntt_ref},
        primes::{PrimeSet, Primes30},
    };

    /// AVX2 NTT followed by AVX2 iNTT is the identity — mirrors the ref test.
    ///
    /// Sweeps up to `n = 2^16` so the by-level phase (`n > CHANGE_MODE_N`),
    /// where the fused radix-4 / level-folding kernels run, is exercised.
    #[test]
    fn ntt_intt_identity_avx2() {
        for log_n in 1..=16usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data, &coeffs);

            let data_orig = data.clone();

            unsafe {
                ntt_avx2::<Primes30>(&fwd, &mut data);
                intt_avx2::<Primes30>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..4 {
                    let orig = data_orig[4 * i + k] % Primes30::Q[k] as u64;
                    let got = data[4 * i + k] % Primes30::Q[k] as u64;
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after AVX2 NTT+iNTT round-trip");
                }
            }
        }
    }

    /// AVX2 NTT-based convolution matches known result.
    ///
    /// a = [1, 2, 0, …], b = [3, 4, 0, …]; a*b mod (X^8+1) = [3, 10, 8, 0, …]
    #[test]
    fn ntt_convolution_avx2() {
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
            ntt_avx2::<Primes30>(&fwd, &mut da);
            ntt_avx2::<Primes30>(&fwd, &mut db);
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
            intt_avx2::<Primes30>(&inv, &mut dc);
        }

        let mut result = vec![0i128; n];
        b_to_znx128_ref::<Primes30>(n, &mut result, &dc);

        let expected: Vec<i128> = [3, 10, 8, 0, 0, 0, 0, 0].to_vec();
        assert_eq!(result, expected, "AVX2 NTT convolution mismatch");
    }

    /// AVX2 NTT output matches reference NTT output.
    ///
    /// Sweeps up to `n = 2^16` so the by-level phase (the only place the fused
    /// kernels differ from the by-block path) is checked bit-for-bit.
    #[test]
    fn ntt_avx2_vs_ref() {
        for log_n in 1..=16usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 13 + 5) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data_avx, &coeffs);
            b_from_znx64_ref::<Primes30>(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx2::<Primes30>(&fwd, &mut data_avx) };
            ntt_ref::<Primes30>(&fwd, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(data_avx[i], data_ref[i], "n={n} idx={i}: NTT AVX2 vs ref mismatch");
            }
        }
    }

    /// AVX2 inverse NTT output matches reference inverse NTT output, bit-for-bit.
    ///
    /// Feeds a forward-transformed vector (via the reference forward NTT, shared
    /// by both sides) into the AVX2 and reference iNTTs and compares the raw
    /// q120b limbs.  Sweeps up to `n = 2^16` to cover the fused by-level path.
    #[test]
    fn intt_avx2_vs_ref() {
        for log_n in 1..=16usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 13 + 5) % 201 - 100).collect();

            // Shared forward transform so both iNTTs see identical input limbs.
            let mut seed = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut seed, &coeffs);
            ntt_ref::<Primes30>(&fwd, &mut seed);

            let mut data_avx = seed.clone();
            let mut data_ref = seed;

            unsafe { intt_avx2::<Primes30>(&inv, &mut data_avx) };
            intt_ref::<Primes30>(&inv, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(data_avx[i], data_ref[i], "n={n} idx={i}: iNTT AVX2 vs ref mismatch");
            }
        }
    }
}
