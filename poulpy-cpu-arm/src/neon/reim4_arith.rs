//! NEON kernels for the Reim4 (4-block) family used by VMP convolution.

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vnegq_f64, vst1q_f64, vsubq_f64,
};

/// Extract a 4-double block from a contiguous `2*rows`-row reim layout.
pub(crate) fn reim4_extract_1blk_contiguous_neon(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    unsafe {
        let mut src_ptr = src.as_ptr().add(blk << 2);
        let mut dst_ptr = dst.as_mut_ptr();
        let step = m;
        for _ in 0..2 * rows {
            let lo = vld1q_f64(src_ptr);
            let hi = vld1q_f64(src_ptr.add(2));
            vst1q_f64(dst_ptr, lo);
            vst1q_f64(dst_ptr.add(2), hi);
            dst_ptr = dst_ptr.add(4);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// Save a 4-double block back into a contiguous `2*rows`-row reim layout.
pub(crate) fn reim4_save_1blk_contiguous_neon(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    unsafe {
        let mut src_ptr = src.as_ptr();
        let mut dst_ptr = dst.as_mut_ptr().add(blk << 2);
        let step = m;
        for _ in 0..2 * rows {
            let lo = vld1q_f64(src_ptr);
            let hi = vld1q_f64(src_ptr.add(2));
            vst1q_f64(dst_ptr, lo);
            vst1q_f64(dst_ptr.add(2), hi);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(4);
        }
    }
}

/// Save a single 4-double block to a reim destination.
pub(crate) fn reim4_save_1blk_neon<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    unsafe {
        let off = blk * 4;
        let src_ptr = src.as_ptr();
        let s0_lo = vld1q_f64(src_ptr);
        let s0_hi = vld1q_f64(src_ptr.add(2));
        let s1_lo = vld1q_f64(src_ptr.add(4));
        let s1_hi = vld1q_f64(src_ptr.add(6));
        let d0_ptr = dst.as_mut_ptr().add(off);
        let d1_ptr = d0_ptr.add(m);
        if OVERWRITE {
            vst1q_f64(d0_ptr, s0_lo);
            vst1q_f64(d0_ptr.add(2), s0_hi);
            vst1q_f64(d1_ptr, s1_lo);
            vst1q_f64(d1_ptr.add(2), s1_hi);
        } else {
            let d0_lo = vld1q_f64(d0_ptr);
            let d0_hi = vld1q_f64(d0_ptr.add(2));
            let d1_lo = vld1q_f64(d1_ptr);
            let d1_hi = vld1q_f64(d1_ptr.add(2));
            vst1q_f64(d0_ptr, vaddq_f64(d0_lo, s0_lo));
            vst1q_f64(d0_ptr.add(2), vaddq_f64(d0_hi, s0_hi));
            vst1q_f64(d1_ptr, vaddq_f64(d1_lo, s1_lo));
            vst1q_f64(d1_ptr.add(2), vaddq_f64(d1_hi, s1_hi));
        }
    }
}

/// Save two 4-double blocks to a reim destination.
pub(crate) fn reim4_save_2blks_neon<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    unsafe {
        let off = blk * 4;
        let src_ptr = src.as_ptr();
        let load = |p: *const f64| -> (float64x2_t, float64x2_t) { (vld1q_f64(p), vld1q_f64(p.add(2))) };
        let (s0_lo, s0_hi) = load(src_ptr);
        let (s1_lo, s1_hi) = load(src_ptr.add(4));
        let (s2_lo, s2_hi) = load(src_ptr.add(8));
        let (s3_lo, s3_hi) = load(src_ptr.add(12));
        let d0_ptr = dst.as_mut_ptr().add(off);
        let d1_ptr = d0_ptr.add(m);
        let d2_ptr = d1_ptr.add(m);
        let d3_ptr = d2_ptr.add(m);
        let store_at = |p: *mut f64, lo: float64x2_t, hi: float64x2_t| {
            vst1q_f64(p, lo);
            vst1q_f64(p.add(2), hi);
        };
        if OVERWRITE {
            store_at(d0_ptr, s0_lo, s0_hi);
            store_at(d1_ptr, s1_lo, s1_hi);
            store_at(d2_ptr, s2_lo, s2_hi);
            store_at(d3_ptr, s3_lo, s3_hi);
        } else {
            let (d0_lo, d0_hi) = load(d0_ptr);
            let (d1_lo, d1_hi) = load(d1_ptr);
            let (d2_lo, d2_hi) = load(d2_ptr);
            let (d3_lo, d3_hi) = load(d3_ptr);
            store_at(d0_ptr, vaddq_f64(d0_lo, s0_lo), vaddq_f64(d0_hi, s0_hi));
            store_at(d1_ptr, vaddq_f64(d1_lo, s1_lo), vaddq_f64(d1_hi, s1_hi));
            store_at(d2_ptr, vaddq_f64(d2_lo, s2_lo), vaddq_f64(d2_hi, s2_hi));
            store_at(d3_ptr, vaddq_f64(d3_lo, s3_lo), vaddq_f64(d3_hi, s3_hi));
        }
    }
}

/// `dst = Σ_row u_row * v_row` (complex), one column.
pub(crate) fn reim4_mat1col_prod_neon(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    debug_assert!(dst.len() >= 8 && u.len() >= nrows * 8 && v.len() >= nrows * 8);
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut re1_lo, mut re1_hi) = (zero, zero);
        let (mut im1_lo, mut im1_hi) = (zero, zero);
        let (mut re2_lo, mut re2_hi) = (zero, zero);
        let (mut im2_lo, mut im2_hi) = (zero, zero);
        let mut u_ptr = u.as_ptr();
        let mut v_ptr = v.as_ptr();
        for _ in 0..nrows {
            let ur_lo = vld1q_f64(u_ptr);
            let ur_hi = vld1q_f64(u_ptr.add(2));
            let ui_lo = vld1q_f64(u_ptr.add(4));
            let ui_hi = vld1q_f64(u_ptr.add(6));
            let vr_lo = vld1q_f64(v_ptr);
            let vr_hi = vld1q_f64(v_ptr.add(2));
            let vi_lo = vld1q_f64(v_ptr.add(4));
            let vi_hi = vld1q_f64(v_ptr.add(6));

            re1_lo = vfmaq_f64(re1_lo, ur_lo, vr_lo);
            re1_hi = vfmaq_f64(re1_hi, ur_hi, vr_hi);
            im1_lo = vfmaq_f64(im1_lo, ur_lo, vi_lo);
            im1_hi = vfmaq_f64(im1_hi, ur_hi, vi_hi);
            re2_lo = vfmaq_f64(re2_lo, ui_lo, vi_lo);
            re2_hi = vfmaq_f64(re2_hi, ui_hi, vi_hi);
            im2_lo = vfmaq_f64(im2_lo, ui_lo, vr_lo);
            im2_hi = vfmaq_f64(im2_hi, ui_hi, vr_hi);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(8);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, vsubq_f64(re1_lo, re2_lo));
        vst1q_f64(d_ptr.add(2), vsubq_f64(re1_hi, re2_hi));
        vst1q_f64(d_ptr.add(4), vaddq_f64(im1_lo, im2_lo));
        vst1q_f64(d_ptr.add(6), vaddq_f64(im1_hi, im2_hi));
    }
}

/// Two-column mat-vec.
pub(crate) fn reim4_mat2cols_prod_neon(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    debug_assert!(dst.len() >= 16 && u.len() >= nrows * 8 && v.len() >= nrows * 16);
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut re_a_pos_lo, mut re_a_pos_hi) = (zero, zero);
        let (mut re_a_neg_lo, mut re_a_neg_hi) = (zero, zero);
        let (mut im_a_lo, mut im_a_hi) = (zero, zero);
        let (mut re_b_pos_lo, mut re_b_pos_hi) = (zero, zero);
        let (mut re_b_neg_lo, mut re_b_neg_hi) = (zero, zero);
        let (mut im_b_lo, mut im_b_hi) = (zero, zero);

        let mut u_ptr = u.as_ptr();
        let mut v_ptr = v.as_ptr();
        for _ in 0..nrows {
            let ur_lo = vld1q_f64(u_ptr);
            let ur_hi = vld1q_f64(u_ptr.add(2));
            let ui_lo = vld1q_f64(u_ptr.add(4));
            let ui_hi = vld1q_f64(u_ptr.add(6));

            let ar_lo = vld1q_f64(v_ptr);
            let ar_hi = vld1q_f64(v_ptr.add(2));
            let ai_lo = vld1q_f64(v_ptr.add(4));
            let ai_hi = vld1q_f64(v_ptr.add(6));
            let br_lo = vld1q_f64(v_ptr.add(8));
            let br_hi = vld1q_f64(v_ptr.add(10));
            let bi_lo = vld1q_f64(v_ptr.add(12));
            let bi_hi = vld1q_f64(v_ptr.add(14));

            // Column a
            re_a_pos_lo = vfmaq_f64(re_a_pos_lo, ur_lo, ar_lo);
            re_a_pos_hi = vfmaq_f64(re_a_pos_hi, ur_hi, ar_hi);
            re_a_neg_lo = vfmaq_f64(re_a_neg_lo, ui_lo, ai_lo);
            re_a_neg_hi = vfmaq_f64(re_a_neg_hi, ui_hi, ai_hi);
            im_a_lo = vfmaq_f64(im_a_lo, ur_lo, ai_lo);
            im_a_hi = vfmaq_f64(im_a_hi, ur_hi, ai_hi);
            im_a_lo = vfmaq_f64(im_a_lo, ui_lo, ar_lo);
            im_a_hi = vfmaq_f64(im_a_hi, ui_hi, ar_hi);
            // Column b
            re_b_pos_lo = vfmaq_f64(re_b_pos_lo, ur_lo, br_lo);
            re_b_pos_hi = vfmaq_f64(re_b_pos_hi, ur_hi, br_hi);
            re_b_neg_lo = vfmaq_f64(re_b_neg_lo, ui_lo, bi_lo);
            re_b_neg_hi = vfmaq_f64(re_b_neg_hi, ui_hi, bi_hi);
            im_b_lo = vfmaq_f64(im_b_lo, ur_lo, bi_lo);
            im_b_hi = vfmaq_f64(im_b_hi, ur_hi, bi_hi);
            im_b_lo = vfmaq_f64(im_b_lo, ui_lo, br_lo);
            im_b_hi = vfmaq_f64(im_b_hi, ui_hi, br_hi);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, vsubq_f64(re_a_pos_lo, re_a_neg_lo));
        vst1q_f64(d_ptr.add(2), vsubq_f64(re_a_pos_hi, re_a_neg_hi));
        vst1q_f64(d_ptr.add(4), im_a_lo);
        vst1q_f64(d_ptr.add(6), im_a_hi);
        vst1q_f64(d_ptr.add(8), vsubq_f64(re_b_pos_lo, re_b_neg_lo));
        vst1q_f64(d_ptr.add(10), vsubq_f64(re_b_pos_hi, re_b_neg_hi));
        vst1q_f64(d_ptr.add(12), im_b_lo);
        vst1q_f64(d_ptr.add(14), im_b_hi);
    }
}

/// Mat-vec for the 2nd column of a packed `[col0, col1]` v-layout.
pub(crate) fn reim4_mat2cols_2ndcol_prod_neon(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    debug_assert!(dst.len() >= 16 && u.len() >= nrows * 8 && v.len() >= nrows * 16);
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let (mut re_pos_lo, mut re_pos_hi) = (zero, zero);
        let (mut re_neg_lo, mut re_neg_hi) = (zero, zero);
        let (mut im_lo, mut im_hi) = (zero, zero);

        let mut u_ptr = u.as_ptr();
        let mut v_ptr = v.as_ptr().add(8); // skip column 0
        for _ in 0..nrows {
            let ur_lo = vld1q_f64(u_ptr);
            let ur_hi = vld1q_f64(u_ptr.add(2));
            let ui_lo = vld1q_f64(u_ptr.add(4));
            let ui_hi = vld1q_f64(u_ptr.add(6));
            let ar_lo = vld1q_f64(v_ptr);
            let ar_hi = vld1q_f64(v_ptr.add(2));
            let ai_lo = vld1q_f64(v_ptr.add(4));
            let ai_hi = vld1q_f64(v_ptr.add(6));

            re_pos_lo = vfmaq_f64(re_pos_lo, ur_lo, ar_lo);
            re_pos_hi = vfmaq_f64(re_pos_hi, ur_hi, ar_hi);
            re_neg_lo = vfmaq_f64(re_neg_lo, ui_lo, ai_lo);
            re_neg_hi = vfmaq_f64(re_neg_hi, ui_hi, ai_hi);
            im_lo = vfmaq_f64(im_lo, ur_lo, ai_lo);
            im_hi = vfmaq_f64(im_hi, ur_hi, ai_hi);
            im_lo = vfmaq_f64(im_lo, ui_lo, ar_lo);
            im_hi = vfmaq_f64(im_hi, ui_hi, ar_hi);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }
        let d_ptr = dst.as_mut_ptr();
        vst1q_f64(d_ptr, vsubq_f64(re_pos_lo, re_neg_lo));
        vst1q_f64(d_ptr.add(2), vsubq_f64(re_pos_hi, re_neg_hi));
        vst1q_f64(d_ptr.add(4), im_lo);
        vst1q_f64(d_ptr.add(6), im_hi);
    }
}

// suppress unused warnings on unused intrinsics if any specific kernel is later
// removed; vfmsq_f64 / vnegq_f64 are kept available for potential future use.
#[allow(dead_code)]
fn _keep() {
    let _ = (vfmsq_f64, vnegq_f64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::fft64::reim4::{
        reim4_extract_1blk_from_reim_contiguous_ref, reim4_save_1blk_to_reim_contiguous_ref, reim4_save_1blk_to_reim_ref,
        reim4_save_2blk_to_reim_ref, reim4_vec_mat1col_product_ref, reim4_vec_mat2cols_2ndcol_product_ref,
        reim4_vec_mat2cols_product_ref,
    };
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0x1234_5678_9abc_def0)
    }

    fn random(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.random::<f64>() * 100.0 - 50.0).collect()
    }

    fn close_enough(got: &[f64], want: &[f64], tag: &str) {
        const REL_TOL: f64 = 1e-12;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            assert!((g - w).abs() / denom < REL_TOL, "{tag}: idx={i} got={g} want={w}");
        }
    }

    #[test]
    fn reim4_extract_1blk_contiguous_neon_matches_ref() {
        let mut r = rng();
        let m = 16usize;
        let rows = 4usize;
        let blk = 2usize;
        let src = random(&mut r, 2 * rows * m);
        let mut got = vec![0f64; 8 * rows];
        let mut want = vec![0f64; 8 * rows];
        reim4_extract_1blk_contiguous_neon(m, rows, blk, &mut got, &src);
        reim4_extract_1blk_from_reim_contiguous_ref(m, rows, blk, &mut want, &src);
        assert_eq!(got, want);
    }

    #[test]
    fn reim4_save_1blk_contiguous_neon_matches_ref() {
        let mut r = rng();
        let m = 16usize;
        let rows = 4usize;
        let blk = 1usize;
        let src = random(&mut r, 8 * rows);
        let mut got = vec![0f64; 2 * rows * m];
        let mut want = vec![0f64; 2 * rows * m];
        reim4_save_1blk_contiguous_neon(m, rows, blk, &mut got, &src);
        reim4_save_1blk_to_reim_contiguous_ref(m, rows, blk, &mut want, &src);
        assert_eq!(got, want);
    }

    #[test]
    fn reim4_save_1blk_neon_matches_ref() {
        let mut r = rng();
        let m = 32usize;
        let blk = 3usize;
        let src = random(&mut r, 8);
        for overwrite in [true, false] {
            let mut got = random(&mut r, 2 * m);
            let mut want = got.clone();
            if overwrite {
                reim4_save_1blk_neon::<true>(m, blk, &mut got, &src);
                reim4_save_1blk_to_reim_ref::<true>(m, blk, &mut want, &src);
            } else {
                reim4_save_1blk_neon::<false>(m, blk, &mut got, &src);
                reim4_save_1blk_to_reim_ref::<false>(m, blk, &mut want, &src);
            }
            assert_eq!(got, want, "overwrite={overwrite}");
        }
    }

    #[test]
    fn reim4_save_2blks_neon_matches_ref() {
        let mut r = rng();
        let m = 32usize;
        let blk = 2usize;
        let src = random(&mut r, 16);
        for overwrite in [true, false] {
            let mut got = random(&mut r, 4 * m);
            let mut want = got.clone();
            if overwrite {
                reim4_save_2blks_neon::<true>(m, blk, &mut got, &src);
                reim4_save_2blk_to_reim_ref::<true>(m, blk, &mut want, &src);
            } else {
                reim4_save_2blks_neon::<false>(m, blk, &mut got, &src);
                reim4_save_2blk_to_reim_ref::<false>(m, blk, &mut want, &src);
            }
            assert_eq!(got, want, "overwrite={overwrite}");
        }
    }

    #[test]
    fn reim4_mat1col_prod_neon_close_to_ref() {
        let mut r = rng();
        for &nrows in &[1usize, 2, 4, 8, 32] {
            let u = random(&mut r, nrows * 8);
            let v = random(&mut r, nrows * 8);
            let mut got = vec![0f64; 8];
            let mut want = vec![0f64; 8];
            reim4_mat1col_prod_neon(nrows, &mut got, &u, &v);
            reim4_vec_mat1col_product_ref(nrows, &mut want, &u, &v);
            close_enough(&got, &want, &format!("mat1col nrows={nrows}"));
        }
    }

    #[test]
    fn reim4_mat2cols_prod_neon_close_to_ref() {
        let mut r = rng();
        for &nrows in &[1usize, 2, 4, 8, 32] {
            let u = random(&mut r, nrows * 8);
            let v = random(&mut r, nrows * 16);
            let mut got = vec![0f64; 16];
            let mut want = vec![0f64; 16];
            reim4_mat2cols_prod_neon(nrows, &mut got, &u, &v);
            reim4_vec_mat2cols_product_ref(nrows, &mut want, &u, &v);
            close_enough(&got, &want, &format!("mat2cols nrows={nrows}"));
        }
    }

    #[test]
    fn reim4_mat2cols_2ndcol_prod_neon_close_to_ref() {
        let mut r = rng();
        for &nrows in &[1usize, 2, 4, 8, 32] {
            let u = random(&mut r, nrows * 8);
            let v = random(&mut r, nrows * 16);
            let mut got = vec![0f64; 16];
            let mut want = vec![0f64; 16];
            reim4_mat2cols_2ndcol_prod_neon(nrows, &mut got, &u, &v);
            reim4_vec_mat2cols_2ndcol_product_ref(nrows, &mut want, &u, &v);
            close_enough(&got, &want, &format!("mat2cols_2ndcol nrows={nrows}"));
        }
    }
}
