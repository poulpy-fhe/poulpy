mod arithmetic_ref;

pub use arithmetic_ref::*;

use crate::reference::fft64::reim::as_arr_mut;

pub trait Reim4BlkMatVec {
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_extract_1blk_from_reim_contiguous_ref(m, rows, blk, dst, src)
    }

    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_contiguous_ref(m, rows, blk, dst, src)
    }

    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src)
    }

    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_2blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src)
    }

    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat1col_product_ref(nrows, dst, u, v)
    }

    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_product_ref(nrows, dst, u, v)
    }

    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_2ndcol_product_ref(nrows, dst, u, v)
    }
}

pub trait Reim4Convolution {
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        reim4_convolution_1coeff_ref(k, dst, a, a_size, b, b_size)
    }

    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        reim4_convolution_2coeffs_ref(k, dst, a, a_size, b, b_size)
    }

    fn reim4_convolution(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        assert!(a_size > 0);
        assert!(b_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            Self::reim4_convolution_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            Self::reim4_convolution_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }
    }

    /// Column-level convolution over all `m/4` blocks into `dst` (limb stride
    /// `2m`, re/im halves `m` apart, block offset `4*blk`).
    #[allow(clippy::too_many_arguments)]
    fn reim4_convolution_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        a: &[f64],
        a_size: usize,
        b: &[f64],
        b_size: usize,
        tmp: &mut [f64],
    ) where
        Self: Reim4BlkMatVec + Sized,
    {
        let a_stride: usize = a_size * 8;
        let b_stride: usize = b_size * 8;
        let mut a_idx: usize = 0;
        let mut b_idx: usize = 0;
        for blk_i in 0..m / 4 {
            Self::reim4_convolution(tmp, min_size, offset, &a[a_idx..], a_size, &b[b_idx..], b_size);
            Self::reim4_save_1blk_contiguous(m, min_size, blk_i, dst, tmp);
            a_idx += a_stride;
            b_idx += b_stride;
        }
    }

    /// Pairwise column-level convolution `(a0 + a1) ⊛ (b0 + b1)`; `tmp` must
    /// additionally hold `8 * (a_size + b_size)` f64 for the summed rows.
    #[allow(clippy::too_many_arguments)]
    fn reim4_convolution_pairwise_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        a0: &[f64],
        a1: &[f64],
        a_size: usize,
        b0: &[f64],
        b1: &[f64],
        b_size: usize,
        tmp: &mut [f64],
    ) where
        Self: Reim4BlkMatVec + crate::reference::fft64::reim::ReimArith + Sized,
    {
        let a_row: usize = a_size * 8;
        let b_row: usize = b_size * 8;
        let (tmp_a, tmp) = tmp.split_at_mut(a_row);
        let (tmp_b, tmp_res) = tmp.split_at_mut(b_row);

        let mut idx_a: usize = 0;
        let mut idx_b: usize = 0;
        for blk_i in 0..m / 4 {
            Self::reim_add(tmp_a, &a0[idx_a..idx_a + a_row], &a1[idx_a..idx_a + a_row]);
            Self::reim_add(tmp_b, &b0[idx_b..idx_b + b_row], &b1[idx_b..idx_b + b_row]);
            Self::reim4_convolution(tmp_res, min_size, offset, tmp_a, a_size, tmp_b, b_size);
            Self::reim4_save_1blk_contiguous(m, min_size, blk_i, dst, tmp_res);
            idx_a += a_row;
            idx_b += b_row;
        }
    }

    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        reim4_convolution_by_real_const_1coeff_ref(k, dst, a, a_size, b)
    }

    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        reim4_convolution_by_real_const_2coeffs_ref(k, dst, a, a_size, b)
    }

    fn reim4_convolution_by_real_const(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64]) {
        assert!(a_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            Self::reim4_convolution_by_real_const_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            Self::reim4_convolution_by_real_const_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }
    }
}
