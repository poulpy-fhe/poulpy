//! NEON matrix-vector dot products for the NTT120 backend.
//!
//! - [`vec_mat1col_product_bbb_neon`] — q120b × q120b → q120b (single column).
//! - [`vec_mat1col_product_bbc_neon`] — q120b × q120c → q120b (single column).
//! - [`vec_mat1col_product_x2_bbc_neon`] — single column, two paired rows.
//! - [`vec_mat2cols_product_x2_bbc_neon`] — two columns, two paired rows.
//! - [`vec_mat1col_product_blkpair_bbc_pm_neon`] — block-pair over prime-major VMP layout.

use core::arch::aarch64::{
    vaddq_u64, vandq_u64, vdupq_n_s64, vdupq_n_u64, vld1q_u64, vmlal_u32, vmovn_u64, vmull_u32, vshlq_u64, vshrq_n_u64,
    vsraq_n_u64, vst1q_u64,
};
use poulpy_cpu_ref::reference::ntt120::{
    mat_vec::{BbbMeta, BbcMeta},
    primes::Primes30,
};

use super::q120::{
    Q120, acc_shr_q120, add_q120, and_q120, load_const, load_q120, mla_epu32_q120, mul_epu32_q120, shr_q120, store_q120,
    zero_q120,
};

/// q120b × q120b → q120b dot product over `ell` elements.
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

            let a = mul_epu32_q120(xl, yl);
            let b = mul_epu32_q120(xl, yh);
            let c = mul_epu32_q120(xh, yl);
            let d = mul_epu32_q120(xh, yh);

            s1 = add_q120(s1, and_q120(a, mask32));
            s2 = acc_shr_q120::<32>(s2, a);
            s2 = add_q120(s2, and_q120(b, mask32));
            s2 = add_q120(s2, and_q120(c, mask32));
            s3 = acc_shr_q120::<32>(s3, b);
            s3 = acc_shr_q120::<32>(s3, c);
            s3 = add_q120(s3, and_q120(d, mask32));
            s4 = acc_shr_q120::<32>(s4, d);

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
        t = mla_epu32_q120(t, s1h, s1h_pow);
        t = mla_epu32_q120(t, s2l, s2l_pow);
        t = mla_epu32_q120(t, s2h, s2h_pow);
        t = mla_epu32_q120(t, s3l, s3l_pow);
        t = mla_epu32_q120(t, s3h, s3h_pow);
        t = mla_epu32_q120(t, s4l, s4l_pow);
        t = mla_epu32_q120(t, s4h, s4h_pow);

        store_q120(res.as_mut_ptr(), t);
    }
}

/// Final-reduction helper shared by all three BBC kernels.
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
        let t = mla_epu32_q120(s_lo, hi_lo, s2l);
        mla_epu32_q120(t, hi_hi, s2h)
    }
}

/// q120b × q120c → q120b inner product (single column).
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
            s2 = acc_shr_q120::<32>(s2, a);
            s2 = acc_shr_q120::<32>(s2, b);

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
            let xa = load_q120(x_ptr);
            let xa_hi = shr_q120::<32>(xa);
            let ya = load_q120(y_ptr);
            let ya_hi = shr_q120::<32>(ya);

            let pa_lo = mul_epu32_q120(xa, ya);
            let pa_hi = mul_epu32_q120(xa_hi, ya_hi);

            s0 = add_q120(s0, and_q120(pa_lo, mask32));
            s0 = add_q120(s0, and_q120(pa_hi, mask32));
            s1 = acc_shr_q120::<32>(s1, pa_lo);
            s1 = acc_shr_q120::<32>(s1, pa_hi);

            let xb = load_q120(x_ptr.add(4));
            let xb_hi = shr_q120::<32>(xb);
            let yb = load_q120(y_ptr.add(4));
            let yb_hi = shr_q120::<32>(yb);

            let pb_lo = mul_epu32_q120(xb, yb);
            let pb_hi = mul_epu32_q120(xb_hi, yb_hi);

            s2 = add_q120(s2, and_q120(pb_lo, mask32));
            s2 = add_q120(s2, and_q120(pb_hi, mask32));
            s3 = acc_shr_q120::<32>(s3, pb_lo);
            s3 = acc_shr_q120::<32>(s3, pb_hi);

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
            s1 = acc_shr_q120::<32>(s1, p0a_lo);
            s1 = acc_shr_q120::<32>(s1, p0a_hi);

            let yc0b = load_q120(y_ptr.add(4));
            let yc0b_hi = shr_q120::<32>(yc0b);
            let p0b_lo = mul_epu32_q120(xb, yc0b);
            let p0b_hi = mul_epu32_q120(xb_hi, yc0b_hi);
            s2 = add_q120(s2, and_q120(p0b_lo, mask32));
            s2 = add_q120(s2, and_q120(p0b_hi, mask32));
            s3 = acc_shr_q120::<32>(s3, p0b_lo);
            s3 = acc_shr_q120::<32>(s3, p0b_hi);

            // Column 1
            let yc1a = load_q120(y_ptr.add(8));
            let yc1a_hi = shr_q120::<32>(yc1a);
            let p1a_lo = mul_epu32_q120(xa, yc1a);
            let p1a_hi = mul_epu32_q120(xa_hi, yc1a_hi);
            s4 = add_q120(s4, and_q120(p1a_lo, mask32));
            s4 = add_q120(s4, and_q120(p1a_hi, mask32));
            s5 = acc_shr_q120::<32>(s5, p1a_lo);
            s5 = acc_shr_q120::<32>(s5, p1a_hi);

            let yc1b = load_q120(y_ptr.add(12));
            let yc1b_hi = shr_q120::<32>(yc1b);
            let p1b_lo = mul_epu32_q120(xb, yc1b);
            let p1b_hi = mul_epu32_q120(xb_hi, yc1b_hi);
            s6 = add_q120(s6, and_q120(p1b_lo, mask32));
            s6 = add_q120(s6, and_q120(p1b_hi, mask32));
            s7 = acc_shr_q120::<32>(s7, p1b_lo);
            s7 = acc_shr_q120::<32>(s7, p1b_hi);

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

/// NEON block-pair inner product over a prime-major VMP layout.
/// `x_pm` contains 4 prime planes. Each plane stores `ell` rows of 4 u64
/// values with lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`. On NEON the
/// 4-u64 row splits into two `uint64x2_t` loads (`xv0` = blk0, `xv1` = blk1).
/// `y_pm` uses the same per-plane/per-row lane order, with each u64 holding a
/// q120c prepared pair for one prime. `y_plane_stride` is the distance, in u64,
/// between consecutive prime planes inside `y_pm`.
/// The output is two standard q120b x2-blocks laid out as 16 u64:
/// `[blk0.c0[4], blk0.c1[4], blk1.c0[4], blk1.c1[4]]`.
pub(crate) unsafe fn vec_mat1col_product_blkpair_bbc_pm_neon(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x_pm: &[u64],
    y_pm: &[u64],
    y_plane_stride: usize,
) {
    debug_assert!(res.len() >= 16);
    debug_assert!(x_pm.len() >= 16 * ell);
    debug_assert!(y_pm.len() >= 3 * y_plane_stride + 4 * ell);

    unsafe {
        let mask32 = vdupq_n_u64(u32::MAX as u64);
        let neg_h2 = vdupq_n_s64(-(meta.h as i64));
        let mask_h2 = vdupq_n_u64((1u64 << meta.h) - 1);
        let x_plane_stride = 4 * ell;
        let mut prime_outputs = [0u64; 16];

        for p in 0..4usize {
            let s2l = vdupq_n_u64(meta.s2l_pow_red[p]);
            let s2h = vdupq_n_u64(meta.s2h_pow_red[p]);
            let s2l_n = vmovn_u64(s2l);
            let s2h_n = vmovn_u64(s2h);
            let x_ptr = x_pm.as_ptr().add(p * x_plane_stride);
            let y_ptr = y_pm.as_ptr().add(p * y_plane_stride);

            let mut s_lo_a = vdupq_n_u64(0);
            let mut s_lo_b = vdupq_n_u64(0);
            let mut s_hi_a = vdupq_n_u64(0);
            let mut s_hi_b = vdupq_n_u64(0);

            for row in 0..ell {
                let off = row * 4;
                let xv_a = vld1q_u64(x_ptr.add(off));
                let xv_b = vld1q_u64(x_ptr.add(off + 2));
                let xl_a = vandq_u64(xv_a, mask32);
                let xl_b = vandq_u64(xv_b, mask32);
                let xh_a = vshrq_n_u64::<32>(xv_a);
                let xh_b = vshrq_n_u64::<32>(xv_b);

                let yv_a = vld1q_u64(y_ptr.add(off));
                let yv_b = vld1q_u64(y_ptr.add(off + 2));
                let y0_a = vandq_u64(yv_a, mask32);
                let y0_b = vandq_u64(yv_b, mask32);
                let y1_a = vshrq_n_u64::<32>(yv_a);
                let y1_b = vshrq_n_u64::<32>(yv_b);

                let prod_lo_a = vmull_u32(vmovn_u64(xl_a), vmovn_u64(y0_a));
                let prod_lo_b = vmull_u32(vmovn_u64(xl_b), vmovn_u64(y0_b));
                let prod_hi_a = vmull_u32(vmovn_u64(xh_a), vmovn_u64(y1_a));
                let prod_hi_b = vmull_u32(vmovn_u64(xh_b), vmovn_u64(y1_b));

                s_lo_a = vaddq_u64(s_lo_a, vandq_u64(prod_lo_a, mask32));
                s_lo_a = vaddq_u64(s_lo_a, vandq_u64(prod_hi_a, mask32));
                s_lo_b = vaddq_u64(s_lo_b, vandq_u64(prod_lo_b, mask32));
                s_lo_b = vaddq_u64(s_lo_b, vandq_u64(prod_hi_b, mask32));
                s_hi_a = vsraq_n_u64::<32>(s_hi_a, prod_lo_a);
                s_hi_a = vsraq_n_u64::<32>(s_hi_a, prod_hi_a);
                s_hi_b = vsraq_n_u64::<32>(s_hi_b, prod_lo_b);
                s_hi_b = vsraq_n_u64::<32>(s_hi_b, prod_hi_b);
            }

            // reduce_bbc: out = s_lo + (s_hi & mask_h2) * s2l + (s_hi >> h2) * s2h
            let hi_lo_a = vandq_u64(s_hi_a, mask_h2);
            let hi_lo_b = vandq_u64(s_hi_b, mask_h2);
            let hi_hi_a = vshlq_u64(s_hi_a, neg_h2);
            let hi_hi_b = vshlq_u64(s_hi_b, neg_h2);
            let t_a = vmlal_u32(s_lo_a, vmovn_u64(hi_lo_a), s2l_n);
            let t_b = vmlal_u32(s_lo_b, vmovn_u64(hi_lo_b), s2l_n);
            let out_a = vmlal_u32(t_a, vmovn_u64(hi_hi_a), s2h_n);
            let out_b = vmlal_u32(t_b, vmovn_u64(hi_hi_b), s2h_n);

            vst1q_u64(prime_outputs.as_mut_ptr().add(4 * p), out_a);
            vst1q_u64(prime_outputs.as_mut_ptr().add(4 * p + 2), out_b);
        }

        // Deinterleave 4 prime planes into the standard 16-u64 q120b x2-block
        // pair layout: [blk0.c0[4 primes], blk0.c1[4], blk1.c0[4], blk1.c1[4]].
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
