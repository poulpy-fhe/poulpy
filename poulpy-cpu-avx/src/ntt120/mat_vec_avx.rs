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

//! AVX2-accelerated Q120 matrix-vector dot products.
//!
//! Rust ports of the BBC (q120b × q120c → q120b) inner-product functions
//! from `q120_arithmetic_avx2.c` in spqlios-arithmetic.
//!
//! # Layout conventions
//!
//! | Format | Bytes/element | AVX2 view |
//! |--------|--------------|-----------|
//! | q120b  | 32 (4 × u64) | one `__m256i` |
//! | q120c  | 32 (8 × u32) | one `__m256i` |
//! | x2-block | 64 (2 × q120b/c) | two `__m256i`s |
//!
//! All three exported functions share the same two-accumulator strategy:
//! accumulate low-32-bit and high-32-bit partial products separately,
//! then collapse with the precomputed `s2l_pow_red` / `s2h_pow_red`
//! constants from [`BbcMeta`].
//!
//! # Functions
//!
//! | AVX2 function | spqlios C equivalent | Trait |
//! |---|---|---|
//! | [`vec_mat1col_product_bbc_avx2`] | `q120_vec_mat1col_product_bbc_avx2` | [`NttMulBbc`] |
//! | [`vec_mat1col_product_x2_bbc_avx2`] | `q120x2_vec_mat1col_product_bbc_avx2` | [`NttMulBbc1ColX2`] |
//! | [`vec_mat2cols_product_x2_bbc_avx2`] | `q120x2_vec_mat2cols_product_bbc_avx2` | [`NttMulBbc2ColsX2`] |

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x,
    _mm256_setzero_si256, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_stream_si256,
};

use poulpy_cpu_ref::reference::ntt120::{mat_vec::BbcMeta, primes::Primes30};

// ─────────────────────────────────────────────────────────────────────────────
// Shared final-reduction helper
// ─────────────────────────────────────────────────────────────────────────────

/// Collapse two accumulator vectors `(s_lo, s_hi)` into a q120b result.
///
/// Computes `res = s_lo + (s_hi & mask_h2) * S2L + (s_hi >> H2) * S2H`,
/// matching the final-reduction step shared by all three BBC functions.
#[inline(always)]
unsafe fn reduce_bbc(s_lo: __m256i, s_hi: __m256i, mask_h2: __m256i, h2: u64, s2l: __m256i, s2h: __m256i) -> __m256i {
    unsafe {
        let h2_count = _mm_cvtsi64_si128(h2 as i64);
        let hi_lo = _mm256_and_si256(s_hi, mask_h2);
        let hi_hi = _mm256_srl_epi64(s_hi, h2_count);
        let t = _mm256_add_epi64(s_lo, _mm256_mul_epu32(hi_lo, s2l));
        _mm256_add_epi64(t, _mm256_mul_epu32(hi_hi, s2h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-column: q120b × q120c → q120b
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 inner product: `res = Σᵢ x[i] · y[i]` in q120b format.
///
/// Port of `q120_vec_mat1col_product_bbc_avx2`.
///
/// - `x`: q120b in u32 view — `ell` elements × 8 u32 (one `__m256i` each).
/// - `y`: q120c — `ell` elements × 8 u32 (one `__m256i` each).
/// - `res`: q120b output — at least 4 u64 (one `__m256i`).
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 8 * ell`, `y.len() >= 8 * ell`, `res.len() >= 4`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_bbc_avx2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mut s1 = _mm256_setzero_si256(); // accumulator for low 32-bit partial products
        let mut s2 = _mm256_setzero_si256(); // accumulator for high 32-bit partial products

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            let xv = _mm256_loadu_si256(x_ptr);
            let xl = _mm256_and_si256(xv, mask32); // low 32 bits of each u64 lane
            let xh = _mm256_srli_epi64::<32>(xv); // high 32 bits shifted down

            let yv = _mm256_loadu_si256(y_ptr);
            let y0 = _mm256_and_si256(yv, mask32); // y_i mod 2^32 (= r_i mod Q)
            let y1 = _mm256_srli_epi64::<32>(yv); // y_i >> 32 (= (r_i * 2^32) mod Q)

            // a = xl * y0  (xl and y0 are both 32-bit, result ≤ 64 bits)
            let a = _mm256_mul_epu32(xl, y0);
            // b = xh * y1
            let b = _mm256_mul_epu32(xh, y1);

            s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_and_si256(b, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(a));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(b));

            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let t = reduce_bbc(s1, s2, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, single column: two q120b × q120c pairs → two q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 x2-block inner product: one column, two paired rows.
///
/// Port of `q120x2_vec_mat1col_product_bbc_avx2`.
///
/// Computes two q120b inner products simultaneously using two interleaved
/// q120b/q120c pairs per step:
/// - `res[0..4]` ← `Σᵢ x_a[i] · y_a[i]`
/// - `res[4..8]` ← `Σᵢ x_b[i] · y_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `y`: x2-block q120c — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `res`: two q120b outputs — at least 8 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 16 * ell`, `res.len() >= 8`.
///
/// `NT_STORE`: when `true`, commit the two q120b outputs with
/// `_mm256_stream_si256`. The caller must then issue one `_mm_sfence`
/// before any subsequent load from `res`, and `res` must be 32-byte aligned.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_x2_bbc_avx2<const NT_STORE: bool>(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // Pair A accumulators
        let mut s0 = _mm256_setzero_si256(); // low sums  for x_a * y_a
        let mut s1 = _mm256_setzero_si256(); // high sums for x_a * y_a
        // Pair B accumulators
        let mut s2 = _mm256_setzero_si256(); // low sums  for x_b * y_b
        let mut s3 = _mm256_setzero_si256(); // high sums for x_b * y_b

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Pair A: x[2i] × y[2i]
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let ya = _mm256_loadu_si256(y_ptr);
            let ya_hi = _mm256_srli_epi64::<32>(ya);

            let prod_a_lo = _mm256_mul_epu32(xa, ya);
            let prod_a_hi = _mm256_mul_epu32(xa_hi, ya_hi);

            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_hi));

            // Pair B: x[2i+1] × y[2i+1]
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);
            let yb = _mm256_loadu_si256(y_ptr.add(1));
            let yb_hi = _mm256_srli_epi64::<32>(yb);

            let prod_b_lo = _mm256_mul_epu32(xb, yb);
            let prod_b_hi = _mm256_mul_epu32(xb_hi, yb_hi);

            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        let out0 = reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        let out1 = reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        if NT_STORE {
            _mm256_stream_si256(res_ptr, out0);
            _mm256_stream_si256(res_ptr.add(1), out1);
        } else {
            _mm256_storeu_si256(res_ptr, out0);
            _mm256_storeu_si256(res_ptr.add(1), out1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-pair, single column, prime-major: four q120b × q120c prime streams
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 block-pair inner product over a prime-major VMP layout.
///
/// `x_pm` contains 4 prime planes. Each plane stores `ell` rows of 4 u64
/// values with lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
///
/// `y_pm` uses the same per-plane/per-row lane order, with each u64 holding a
/// q120c prepared pair for one prime. `y_plane_stride` is the distance, in u64,
/// between consecutive prime planes inside `y_pm`.
///
/// The output is two standard q120b x2-blocks laid out as 16 u64:
/// `[blk0.c0[4], blk0.c1[4], blk1.c0[4], blk1.c1[4]]`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_blkpair_bbc_pm_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x_pm: &[u64],
    y_pm: &[u64],
    y_plane_stride: usize,
) {
    unsafe {
        debug_assert!(res.len() >= 16);
        debug_assert!(x_pm.len() >= 16 * ell);
        debug_assert!(y_pm.len() >= 3 * y_plane_stride + 4 * ell);

        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let x_plane_stride = 4 * ell;
        let mut prime_outputs = [0u64; 16];

        for p in 0..4usize {
            let s2l_pow_red = _mm256_set1_epi64x(meta.s2l_pow_red[p] as i64);
            let s2h_pow_red = _mm256_set1_epi64x(meta.s2h_pow_red[p] as i64);
            let x_ptr = x_pm.as_ptr().add(p * x_plane_stride) as *const __m256i;
            let y_ptr = y_pm.as_ptr().add(p * y_plane_stride) as *const __m256i;

            let mut s_lo = _mm256_setzero_si256();
            let mut s_hi = _mm256_setzero_si256();

            for row in 0..ell {
                let xv = _mm256_loadu_si256(x_ptr.add(row));
                let xl = _mm256_and_si256(xv, mask32);
                let xh = _mm256_srli_epi64::<32>(xv);

                let yv = _mm256_loadu_si256(y_ptr.add(row));
                let y0 = _mm256_and_si256(yv, mask32);
                let y1 = _mm256_srli_epi64::<32>(yv);

                let prod_lo = _mm256_mul_epu32(xl, y0);
                let prod_hi = _mm256_mul_epu32(xh, y1);

                s_lo = _mm256_add_epi64(s_lo, _mm256_and_si256(prod_lo, mask32));
                s_lo = _mm256_add_epi64(s_lo, _mm256_and_si256(prod_hi, mask32));
                s_hi = _mm256_add_epi64(s_hi, _mm256_srli_epi64::<32>(prod_lo));
                s_hi = _mm256_add_epi64(s_hi, _mm256_srli_epi64::<32>(prod_hi));
            }

            let out = reduce_bbc(s_lo, s_hi, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
            _mm256_storeu_si256(prime_outputs.as_mut_ptr().add(4 * p) as *mut __m256i, out);
        }

        res[0] = prime_outputs[0];
        res[1] = prime_outputs[4];
        res[2] = prime_outputs[8];
        res[3] = prime_outputs[12];
        res[4] = prime_outputs[1];
        res[5] = prime_outputs[5];
        res[6] = prime_outputs[9];
        res[7] = prime_outputs[13];
        res[8] = prime_outputs[2];
        res[9] = prime_outputs[6];
        res[10] = prime_outputs[10];
        res[11] = prime_outputs[14];
        res[12] = prime_outputs[3];
        res[13] = prime_outputs[7];
        res[14] = prime_outputs[11];
        res[15] = prime_outputs[15];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, two columns: two q120b × four q120c pairs → four q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 x2-block inner product: two columns simultaneously.
///
/// Port of `q120x2_vec_mat2cols_product_bbc_avx2`.
///
/// Computes four q120b inner products (two x2-block rows × two matrix columns):
/// - `res[0..4]`   ← `Σᵢ x_a[i] · y_col0_a[i]`
/// - `res[4..8]`   ← `Σᵢ x_b[i] · y_col0_b[i]`
/// - `res[8..12]`  ← `Σᵢ x_a[i] · y_col1_a[i]`
/// - `res[12..16]` ← `Σᵢ x_b[i] · y_col1_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` × 16 u32 (two `__m256i`s per step).
/// - `y`: two paired x2-block q120c columns — `ell` × 32 u32 (four `__m256i`s per step):
///   `[col0_a, col0_b, col1_a, col1_b]` per element.
/// - `res`: four q120b outputs — at least 16 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 32 * ell`, `res.len() >= 16`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat2cols_product_x2_bbc_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);

        // col 0, pair A
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        // col 0, pair B
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();
        // col 1, pair A
        let mut s4 = _mm256_setzero_si256();
        let mut s5 = _mm256_setzero_si256();
        // col 1, pair B
        let mut s6 = _mm256_setzero_si256();
        let mut s7 = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Load x pair
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);

            // ── Column 0 ──
            // pair A: xa × y_col0_a
            let yc0a = _mm256_loadu_si256(y_ptr);
            let yc0a_hi = _mm256_srli_epi64::<32>(yc0a);
            let p0a_lo = _mm256_mul_epu32(xa, yc0a);
            let p0a_hi = _mm256_mul_epu32(xa_hi, yc0a_hi);
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_hi));

            // pair B: xb × y_col0_b
            let yc0b = _mm256_loadu_si256(y_ptr.add(1));
            let yc0b_hi = _mm256_srli_epi64::<32>(yc0b);
            let p0b_lo = _mm256_mul_epu32(xb, yc0b);
            let p0b_hi = _mm256_mul_epu32(xb_hi, yc0b_hi);
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_hi));

            // ── Column 1 ──
            // pair A: xa × y_col1_a
            let yc1a = _mm256_loadu_si256(y_ptr.add(2));
            let yc1a_hi = _mm256_srli_epi64::<32>(yc1a);
            let p1a_lo = _mm256_mul_epu32(xa, yc1a);
            let p1a_hi = _mm256_mul_epu32(xa_hi, yc1a_hi);
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_lo, mask32));
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_hi, mask32));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_lo));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_hi));

            // pair B: xb × y_col1_b
            let yc1b = _mm256_loadu_si256(y_ptr.add(3));
            let yc1b_hi = _mm256_srli_epi64::<32>(yc1b);
            let p1b_lo = _mm256_mul_epu32(xb, yc1b);
            let p1b_hi = _mm256_mul_epu32(xb_hi, yc1b_hi);
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_lo, mask32));
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_hi, mask32));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_lo));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(4);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(res_ptr, reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(1), reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(2), reduce_bbc(s4, s5, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(3), reduce_bbc(s6, s7, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use super::*;
    use bytemuck::cast_slice;
    use core::arch::x86_64::_mm256_set_epi64x;
    use poulpy_cpu_ref::reference::ntt120::{
        arithmetic::{b_from_znx64_ref, c_from_b_ref},
        mat_vec::{BbcMeta, vec_mat1col_product_bbc_ref, vec_mat1col_product_x2_bbc_ref, vec_mat2cols_product_x2_bbc_ref},
        primes::Primes30,
    };

    /// Cast a q120b `[u64]` slice to `[u32]` for bbc functions.
    /// Each u64 limb fits in u32 (< 2^30), so upper 32 bits are 0.
    fn b_to_u32(b: &[u64]) -> Vec<u32> {
        b.iter().flat_map(|&v| [v as u32, (v >> 32) as u32]).collect()
    }

    /// Build a q120b slice (as u32) from small i64 coefficients.
    fn make_q120b_u32(ell: usize, n: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..ell * n).map(|i| (i as i64 * seed + 1) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * ell * n];
        b_from_znx64_ref::<Primes30>(ell * n, &mut b, &coeffs);
        b_to_u32(&b)
    }

    /// Build a q120c slice (as u32) from a q120b u64 slice.
    fn make_q120c_u32(ell: usize, n: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..ell * n).map(|i| (i as i64 * seed + 2) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * ell * n];
        b_from_znx64_ref::<Primes30>(ell * n, &mut b, &coeffs);
        let mut c = vec![0u32; 8 * ell * n];
        c_from_b_ref::<Primes30>(ell * n, &mut c, &b);
        c
    }

    /// AVX2 `vec_mat1col_product_bbc` matches reference (single column, single output).
    #[test]
    fn vec_mat1col_product_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize; // one element per row
        let meta = BbcMeta::<Primes30>::new();

        let x = make_q120b_u32(ell, n, 7); // ell × 8 u32
        let y = make_q120c_u32(ell, n, 13); // ell × 8 u32

        let mut res_avx = vec![0u64; 4];
        let mut res_ref = vec![0u64; 4];

        unsafe { vec_mat1col_product_bbc_avx2(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat1col_product_bbc_ref::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_bbc: AVX2 vs ref mismatch");
    }

    /// AVX2 `vec_mat1col_product_x2_bbc` matches reference (single column, two simultaneous outputs).
    #[test]
    fn vec_mat1col_product_x2_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize;
        let meta = BbcMeta::<Primes30>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, n, 5);
            let b = make_q120b_u32(ell, n, 11);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };
        // y: 2 interleaved q120c (16 u32 per row)
        let y: Vec<u32> = {
            let a = make_q120c_u32(ell, n, 3);
            let b = make_q120c_u32(ell, n, 17);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };

        let mut res_avx = vec![0u64; 8];
        let mut res_ref = vec![0u64; 8];

        unsafe { vec_mat1col_product_x2_bbc_avx2::<false>(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat1col_product_x2_bbc_ref::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_x2_bbc: AVX2 vs ref mismatch");
    }

    /// AVX2 `vec_mat2cols_product_x2_bbc` matches reference (two columns, two simultaneous outputs).
    #[test]
    fn vec_mat2cols_product_x2_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize;
        let meta = BbcMeta::<Primes30>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, n, 7);
            let b = make_q120b_u32(ell, n, 19);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };
        // y: 4 interleaved q120c (32 u32 per row: col0_a, col0_b, col1_a, col1_b)
        let y: Vec<u32> = {
            let c0a = make_q120c_u32(ell, n, 2);
            let c0b = make_q120c_u32(ell, n, 9);
            let c1a = make_q120c_u32(ell, n, 23);
            let c1b = make_q120c_u32(ell, n, 31);
            (0..ell)
                .flat_map(|i| {
                    c0a[8 * i..8 * i + 8]
                        .iter()
                        .chain(c0b[8 * i..8 * i + 8].iter())
                        .chain(c1a[8 * i..8 * i + 8].iter())
                        .chain(c1b[8 * i..8 * i + 8].iter())
                        .cloned()
                })
                .collect()
        };

        let mut res_avx = vec![0u64; 16];
        let mut res_ref = vec![0u64; 16];

        unsafe { vec_mat2cols_product_x2_bbc_avx2(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat2cols_product_x2_bbc_ref::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat2cols_product_x2_bbc: AVX2 vs ref mismatch");
    }

    #[test]
    fn vec_mat1col_product_blkpair_bbc_pm_avx2_vs_ref() {
        let ell = 8usize;
        let meta = BbcMeta::<Primes30>::new();

        let coeffs_x: Vec<i64> = (0..ell * 4).map(|i| (i as i64 * 7 + 5) % 53 + 1).collect();
        let coeffs_y: Vec<i64> = (0..ell * 4).map(|i| (i as i64 * 11 + 3) % 59 + 1).collect();

        let mut x_b = vec![0u64; 16 * ell];
        let mut y_b = vec![0u64; 16 * ell];
        b_from_znx64_ref::<Primes30>(ell * 4, &mut x_b, &coeffs_x);
        b_from_znx64_ref::<Primes30>(ell * 4, &mut y_b, &coeffs_y);

        let mut y_c = vec![0u32; 32 * ell];
        c_from_b_ref::<Primes30>(ell * 4, &mut y_c, &y_b);
        let y_c_u64: &[u64] = cast_slice(&y_c);
        let x_b_u32 = b_to_u32(&x_b);

        let x_pm: Vec<u64> = {
            let plane_stride = 4 * ell;
            let mut out = vec![0u64; 4 * plane_stride];
            for row in 0..ell {
                let row_base = row * 16;
                for p in 0..4usize {
                    let dst = out.as_mut_ptr().wrapping_add(p * plane_stride + row * 4) as *mut __m256i;
                    unsafe {
                        _mm256_storeu_si256(
                            dst,
                            _mm256_set_epi64x(
                                x_b[row_base + 12 + p] as i64,
                                x_b[row_base + 8 + p] as i64,
                                x_b[row_base + 4 + p] as i64,
                                x_b[row_base + p] as i64,
                            ),
                        );
                    }
                }
            }
            out
        };

        let y_pm: Vec<u64> = {
            let plane_stride = 4 * ell;
            let mut out = vec![0u64; 4 * plane_stride];
            for row in 0..ell {
                let row_base = row * 16;
                for p in 0..4usize {
                    let dst = out.as_mut_ptr().wrapping_add(p * plane_stride + row * 4) as *mut __m256i;
                    unsafe {
                        _mm256_storeu_si256(
                            dst,
                            _mm256_set_epi64x(
                                y_c_u64[row_base + 12 + p] as i64,
                                y_c_u64[row_base + 8 + p] as i64,
                                y_c_u64[row_base + 4 + p] as i64,
                                y_c_u64[row_base + p] as i64,
                            ),
                        );
                    }
                }
            }
            out
        };

        let mut res_avx = vec![0u64; 16];
        unsafe { vec_mat1col_product_blkpair_bbc_pm_avx2(&meta, ell, &mut res_avx, &x_pm, &y_pm, 4 * ell) };

        let mut res_ref = vec![0u64; 16];
        for coeff in 0..4usize {
            let x_coeff: Vec<u32> = (0..ell)
                .flat_map(|row| x_b_u32[row * 32 + coeff * 8..row * 32 + (coeff + 1) * 8].iter().copied())
                .collect();
            let y_coeff: Vec<u32> = (0..ell)
                .flat_map(|row| y_c[row * 32 + coeff * 8..row * 32 + (coeff + 1) * 8].iter().copied())
                .collect();
            vec_mat1col_product_bbc_ref::<Primes30>(&meta, ell, &mut res_ref[4 * coeff..4 * (coeff + 1)], &x_coeff, &y_coeff);
        }

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_blkpair_bbc_pm: AVX2 vs ref mismatch");
    }
}
