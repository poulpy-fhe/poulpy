//! NEON matrix-vector dot products for the NTT120 backend.
//!
//! Mirrors `poulpy-cpu-avx/src/ntt120/{arithmetic_avx.rs::vec_mat1col_product_bbb_avx2,
//! mat_vec_avx.rs}`. Four kernels:
//!
//! - [`vec_mat1col_product_bbb_neon`] — q120b × q120b → q120b (single column).
//! - [`vec_mat1col_product_bbc_neon`] — q120b × q120c → q120b (single column).
//! - [`vec_mat1col_product_x2_bbc_neon`] — single column, two paired rows.
//! - [`vec_mat2cols_product_x2_bbc_neon`] — two columns, two paired rows.
//!
//! All four follow the AVX two-accumulator (or four-accumulator for BBB) strategy:
//! sum low-32-bit and high-32-bit partial products separately, collapse with
//! the precomputed `s*l_pow_red` / `s*h_pow_red` constants from `BbcMeta` /
//! `BbbMeta`. NEON has 2 × u64 lanes per register vs AVX's 4, so each q120
//! coefficient is a `Q120` pair (`lo`, `hi`); per-iteration arithmetic
//! roughly doubles in instruction count compared to AVX.
//!
//! **Status**: bit-exactness against `vec_mat*_ref` is asserted by the HAL
//! `cross_backend_test_suite!` tests, which currently pass under
//! `qemu-aarch64-static`. Re-run on a real aarch64 host before relying on
//! performance characteristics.

use core::arch::aarch64::{vdupq_n_s64, vdupq_n_u64, vshlq_u64};
use poulpy_cpu_ref::reference::ntt120::{
    mat_vec::{BbbMeta, BbcMeta},
    primes::Primes30,
};

use super::q120::{Q120, add_q120, and_q120, load_const, load_q120, mul_epu32_q120, shr_q120, store_q120, zero_q120};

// ─────────────────────────────────────────────────────────────────────────────
// vec_mat1col_product_bbb_neon
// ─────────────────────────────────────────────────────────────────────────────

/// q120b × q120b → q120b dot product over `ell` elements.
/// Mirrors `vec_mat1col_product_bbb_avx2` at `arithmetic_avx.rs:633`.
pub(crate) fn vec_mat1col_product_bbb_neon(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 4 * ell);
    assert!(y.len() >= 4 * ell);
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        let mut s1 = zero_q120();
        let mut s2 = zero_q120();
        let mut s3 = zero_q120();
        let mut s4 = zero_q120();

        let mut x_ptr = x.as_ptr();
        let mut y_ptr = y.as_ptr();

        for _ in 0..ell {
            let xv = load_q120(x_ptr);
            let xl = and_q120(xv, mask32);
            let xh = shr_q120::<32>(xv);

            let yv = load_q120(y_ptr);
            let yl = and_q120(yv, mask32);
            let yh = shr_q120::<32>(yv);

            // Four 32×32→64 cross-products
            let a = mul_epu32_q120(xl, yl);
            let b = mul_epu32_q120(xl, yh);
            let c = mul_epu32_q120(xh, yl);
            let d = mul_epu32_q120(xh, yh);

            // Bin accumulation (matches scalar reference)
            s1 = add_q120(s1, and_q120(a, mask32));
            s2 = add_q120(s2, shr_q120::<32>(a));
            s2 = add_q120(s2, and_q120(b, mask32));
            s2 = add_q120(s2, and_q120(c, mask32));
            s3 = add_q120(s3, shr_q120::<32>(b));
            s3 = add_q120(s3, shr_q120::<32>(c));
            s3 = add_q120(s3, and_q120(d, mask32));
            s4 = add_q120(s4, shr_q120::<32>(d));

            x_ptr = x_ptr.add(4);
            y_ptr = y_ptr.add(4);
        }

        // Final reduction with BbbMeta constants (variable shift by `meta.h`).
        let h2 = meta.h;
        let neg_h2 = vdupq_n_s64(-(h2 as i64));
        let mask_h2_v = vdupq_n_u64((1u64 << h2) - 1);
        let mask_h2 = Q120 {
            lo: mask_h2_v,
            hi: mask_h2_v,
        };
        let s1h_pow_v = vdupq_n_u64(meta.s1h_pow_red);
        let s1h_pow = Q120 {
            lo: s1h_pow_v,
            hi: s1h_pow_v,
        };
        let s2l_pow = load_const(&meta.s2l_pow_red);
        let s2h_pow = load_const(&meta.s2h_pow_red);
        let s3l_pow = load_const(&meta.s3l_pow_red);
        let s3h_pow = load_const(&meta.s3h_pow_red);
        let s4l_pow = load_const(&meta.s4l_pow_red);
        let s4h_pow = load_const(&meta.s4h_pow_red);

        // Split each sX into low-h2 and high-h2 bits via variable right shift.
        let split = |s: Q120| -> (Q120, Q120) {
            let lo = and_q120(s, mask_h2);
            let hi = Q120 {
                lo: vshlq_u64(s.lo, neg_h2),
                hi: vshlq_u64(s.hi, neg_h2),
            };
            (lo, hi)
        };
        let (s1l, s1h) = split(s1);
        let (s2l, s2h) = split(s2);
        let (s3l, s3h) = split(s3);
        let (s4l, s4h) = split(s4);

        // t = s1l + s1h*s1h_pow + s2l*s2l_pow + s2h*s2h_pow
        //       + s3l*s3l_pow + s3h*s3h_pow + s4l*s4l_pow + s4h*s4h_pow
        let mut t = s1l;
        t = add_q120(t, mul_epu32_q120(s1h, s1h_pow));
        t = add_q120(t, mul_epu32_q120(s2l, s2l_pow));
        t = add_q120(t, mul_epu32_q120(s2h, s2h_pow));
        t = add_q120(t, mul_epu32_q120(s3l, s3l_pow));
        t = add_q120(t, mul_epu32_q120(s3h, s3h_pow));
        t = add_q120(t, mul_epu32_q120(s4l, s4l_pow));
        t = add_q120(t, mul_epu32_q120(s4h, s4h_pow));

        store_q120(res.as_mut_ptr(), t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BBC kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Final-reduction helper shared by all three BBC kernels.
/// Mirrors `reduce_bbc` at `mat_vec_avx.rs:60`.
#[inline(always)]
unsafe fn reduce_bbc_neon(
    s_lo: Q120,
    s_hi: Q120,
    mask_h2: Q120,
    neg_h2: core::arch::aarch64::int64x2_t,
    s2l: Q120,
    s2h: Q120,
) -> Q120 {
    unsafe {
        let hi_lo = and_q120(s_hi, mask_h2);
        let hi_hi = Q120 {
            lo: vshlq_u64(s_hi.lo, neg_h2),
            hi: vshlq_u64(s_hi.hi, neg_h2),
        };
        let t = add_q120(s_lo, mul_epu32_q120(hi_lo, s2l));
        add_q120(t, mul_epu32_q120(hi_hi, s2h))
    }
}

/// q120b × q120c → q120b inner product (single column).
/// Mirrors `vec_mat1col_product_bbc_avx2` at `mat_vec_avx.rs:87`.
pub(crate) fn vec_mat1col_product_bbc_neon(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 8 * ell);
    assert!(y.len() >= 8 * ell);
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        let mut s1 = zero_q120();
        let mut s2 = zero_q120();

        let mut x_ptr = x.as_ptr() as *const u64;
        let mut y_ptr = y.as_ptr() as *const u64;

        for _ in 0..ell {
            let xv = load_q120(x_ptr);
            let xl = and_q120(xv, mask32);
            let xh = shr_q120::<32>(xv);

            let yv = load_q120(y_ptr);
            let y0 = and_q120(yv, mask32);
            let y1 = shr_q120::<32>(yv);

            let a = mul_epu32_q120(xl, y0);
            let b = mul_epu32_q120(xh, y1);

            s1 = add_q120(s1, and_q120(a, mask32));
            s1 = add_q120(s1, and_q120(b, mask32));
            s2 = add_q120(s2, shr_q120::<32>(a));
            s2 = add_q120(s2, shr_q120::<32>(b));

            x_ptr = x_ptr.add(4);
            y_ptr = y_ptr.add(4);
        }

        let neg_h2 = vdupq_n_s64(-(meta.h as i64));
        let mask_h2_v = vdupq_n_u64((1u64 << meta.h) - 1);
        let mask_h2 = Q120 {
            lo: mask_h2_v,
            hi: mask_h2_v,
        };
        let s2l = load_const(&meta.s2l_pow_red);
        let s2h = load_const(&meta.s2h_pow_red);

        let t = reduce_bbc_neon(s1, s2, mask_h2, neg_h2, s2l, s2h);
        store_q120(res.as_mut_ptr(), t);
    }
}

/// x2-block, single column: two paired q120b × q120c inner products.
/// Mirrors `vec_mat1col_product_x2_bbc_avx2` at `mat_vec_avx.rs:150`.
pub(crate) fn vec_mat1col_product_x2_bbc_neon(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 8);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 16 * ell);
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        let mut s0 = zero_q120();
        let mut s1 = zero_q120();
        let mut s2 = zero_q120();
        let mut s3 = zero_q120();

        let mut x_ptr = x.as_ptr() as *const u64;
        let mut y_ptr = y.as_ptr() as *const u64;

        for _ in 0..ell {
            // Pair A: x[2i] × y[2i]
            let xa = load_q120(x_ptr);
            let xa_hi = shr_q120::<32>(xa);
            let ya = load_q120(y_ptr);
            let ya_hi = shr_q120::<32>(ya);

            let pa_lo = mul_epu32_q120(xa, ya);
            let pa_hi = mul_epu32_q120(xa_hi, ya_hi);

            s0 = add_q120(s0, and_q120(pa_lo, mask32));
            s0 = add_q120(s0, and_q120(pa_hi, mask32));
            s1 = add_q120(s1, shr_q120::<32>(pa_lo));
            s1 = add_q120(s1, shr_q120::<32>(pa_hi));

            // Pair B: x[2i+1] × y[2i+1]
            let xb = load_q120(x_ptr.add(4));
            let xb_hi = shr_q120::<32>(xb);
            let yb = load_q120(y_ptr.add(4));
            let yb_hi = shr_q120::<32>(yb);

            let pb_lo = mul_epu32_q120(xb, yb);
            let pb_hi = mul_epu32_q120(xb_hi, yb_hi);

            s2 = add_q120(s2, and_q120(pb_lo, mask32));
            s2 = add_q120(s2, and_q120(pb_hi, mask32));
            s3 = add_q120(s3, shr_q120::<32>(pb_lo));
            s3 = add_q120(s3, shr_q120::<32>(pb_hi));

            x_ptr = x_ptr.add(8);
            y_ptr = y_ptr.add(8);
        }

        let neg_h2 = vdupq_n_s64(-(meta.h as i64));
        let mask_h2_v = vdupq_n_u64((1u64 << meta.h) - 1);
        let mask_h2 = Q120 {
            lo: mask_h2_v,
            hi: mask_h2_v,
        };
        let s2l = load_const(&meta.s2l_pow_red);
        let s2h = load_const(&meta.s2h_pow_red);

        let res_ptr = res.as_mut_ptr();
        store_q120(res_ptr, reduce_bbc_neon(s0, s1, mask_h2, neg_h2, s2l, s2h));
        store_q120(res_ptr.add(4), reduce_bbc_neon(s2, s3, mask_h2, neg_h2, s2l, s2h));
    }
}

/// x2-block, two columns: four paired inner products.
/// Mirrors `vec_mat2cols_product_x2_bbc_avx2` at `mat_vec_avx.rs:236`.
pub(crate) fn vec_mat2cols_product_x2_bbc_neon(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 16);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 32 * ell);
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        let mut s0 = zero_q120();
        let mut s1 = zero_q120();
        let mut s2 = zero_q120();
        let mut s3 = zero_q120();
        let mut s4 = zero_q120();
        let mut s5 = zero_q120();
        let mut s6 = zero_q120();
        let mut s7 = zero_q120();

        let mut x_ptr = x.as_ptr() as *const u64;
        let mut y_ptr = y.as_ptr() as *const u64;

        for _ in 0..ell {
            // Load x pair
            let xa = load_q120(x_ptr);
            let xa_hi = shr_q120::<32>(xa);
            let xb = load_q120(x_ptr.add(4));
            let xb_hi = shr_q120::<32>(xb);

            // Column 0
            let yc0a = load_q120(y_ptr);
            let yc0a_hi = shr_q120::<32>(yc0a);
            let p0a_lo = mul_epu32_q120(xa, yc0a);
            let p0a_hi = mul_epu32_q120(xa_hi, yc0a_hi);
            s0 = add_q120(s0, and_q120(p0a_lo, mask32));
            s0 = add_q120(s0, and_q120(p0a_hi, mask32));
            s1 = add_q120(s1, shr_q120::<32>(p0a_lo));
            s1 = add_q120(s1, shr_q120::<32>(p0a_hi));

            let yc0b = load_q120(y_ptr.add(4));
            let yc0b_hi = shr_q120::<32>(yc0b);
            let p0b_lo = mul_epu32_q120(xb, yc0b);
            let p0b_hi = mul_epu32_q120(xb_hi, yc0b_hi);
            s2 = add_q120(s2, and_q120(p0b_lo, mask32));
            s2 = add_q120(s2, and_q120(p0b_hi, mask32));
            s3 = add_q120(s3, shr_q120::<32>(p0b_lo));
            s3 = add_q120(s3, shr_q120::<32>(p0b_hi));

            // Column 1
            let yc1a = load_q120(y_ptr.add(8));
            let yc1a_hi = shr_q120::<32>(yc1a);
            let p1a_lo = mul_epu32_q120(xa, yc1a);
            let p1a_hi = mul_epu32_q120(xa_hi, yc1a_hi);
            s4 = add_q120(s4, and_q120(p1a_lo, mask32));
            s4 = add_q120(s4, and_q120(p1a_hi, mask32));
            s5 = add_q120(s5, shr_q120::<32>(p1a_lo));
            s5 = add_q120(s5, shr_q120::<32>(p1a_hi));

            let yc1b = load_q120(y_ptr.add(12));
            let yc1b_hi = shr_q120::<32>(yc1b);
            let p1b_lo = mul_epu32_q120(xb, yc1b);
            let p1b_hi = mul_epu32_q120(xb_hi, yc1b_hi);
            s6 = add_q120(s6, and_q120(p1b_lo, mask32));
            s6 = add_q120(s6, and_q120(p1b_hi, mask32));
            s7 = add_q120(s7, shr_q120::<32>(p1b_lo));
            s7 = add_q120(s7, shr_q120::<32>(p1b_hi));

            x_ptr = x_ptr.add(8);
            y_ptr = y_ptr.add(16);
        }

        let neg_h2 = vdupq_n_s64(-(meta.h as i64));
        let mask_h2_v = vdupq_n_u64((1u64 << meta.h) - 1);
        let mask_h2 = Q120 {
            lo: mask_h2_v,
            hi: mask_h2_v,
        };
        let s2l = load_const(&meta.s2l_pow_red);
        let s2h = load_const(&meta.s2h_pow_red);

        let res_ptr = res.as_mut_ptr();
        store_q120(res_ptr, reduce_bbc_neon(s0, s1, mask_h2, neg_h2, s2l, s2h));
        store_q120(res_ptr.add(4), reduce_bbc_neon(s2, s3, mask_h2, neg_h2, s2l, s2h));
        store_q120(res_ptr.add(8), reduce_bbc_neon(s4, s5, mask_h2, neg_h2, s2l, s2h));
        store_q120(res_ptr.add(12), reduce_bbc_neon(s6, s7, mask_h2, neg_h2, s2l, s2h));
    }
}
