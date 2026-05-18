use std::{
    num::Wrapping,
    ops::{Add, Mul},
};

use crate::reference::coeff_mat::{coeff_mat1col_product, coeff_mat2cols_product};
use poulpy_hal::layouts::{
    Backend, CoeffMatPMatBackendMut, CoeffMatPMatBackendRef, HostDataMut, HostDataRef, Module, ScratchArena, VecZnxBackendRef,
    VecZnxBigBackendMut, ZnxView, ZnxViewMut,
};

#[doc(hidden)]
pub trait CoeffMatPMatDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    fn coeff_mat_prepare_tmp_bytes_default(
        _module: &Module<BE>,
        _rows: usize,
        _cols_in: usize,
        _cols_out: usize,
        _size: usize,
    ) -> usize {
        0
    }

    fn coeff_mat_prepare_default(
        _module: &Module<BE>,
        res: &mut CoeffMatPMatBackendMut<'_, BE>,
        matrix: &VecZnxBackendRef<'_, BE>,
        _scratch: &mut ScratchArena<'_, BE>,
    ) {
        assert_eq!(res.rows(), 1, "coeff_mat_prepare currently expects one input-column group");
        assert_eq!(res.cols_in(), 1, "coeff_mat_prepare currently expects one input column");
        assert_eq!(res.cols_out(), matrix.cols(), "coeff_mat_prepare: output column mismatch");
        assert_eq!(res.size(), matrix.size(), "coeff_mat_prepare: limb count mismatch");
        assert_eq!(res.n(), matrix.n(), "coeff_mat_prepare: ring degree mismatch");

        res.raw_mut().fill(0);
        for limb in 0..matrix.size() {
            for out_col in 0..matrix.cols() {
                let pmat_col = limb * res.cols_out() + out_col;
                let src = matrix.at(out_col, limb);
                for (coeff, &value) in src.iter().enumerate() {
                    res.set_packed(0, pmat_col, coeff, value);
                }
            }
        }
    }

    fn coeff_mat_apply_big_tmp_bytes_default(_module: &Module<BE>, _rows_in: usize, _rows_out: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn coeff_mat_apply_big_default(
        _module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_limb: usize,
        pmat: &CoeffMatPMatBackendRef<'_, BE>,
        pmat_limb: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_limb: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE::ScalarBig: Copy + From<i64>,
        Wrapping<BE::ScalarBig>: Add<Output = Wrapping<BE::ScalarBig>> + Mul<Output = Wrapping<BE::ScalarBig>>,
    {
        assert!(rows_in <= a.n(), "coeff_mat_apply_big: rows_in exceeds input degree");
        assert!(rows_in <= pmat.n(), "coeff_mat_apply_big: rows_in exceeds prepared degree");
        assert!(
            rows_out <= pmat.cols_out(),
            "coeff_mat_apply_big: rows_out exceeds prepared columns"
        );
        assert!(rows_out <= res.cols(), "coeff_mat_apply_big: rows_out exceeds result columns");
        assert!(res_limb < res.size(), "coeff_mat_apply_big: result limb out of bounds");
        assert!(pmat_limb < pmat.size(), "coeff_mat_apply_big: prepared limb out of bounds");

        for out_row in 0..rows_out {
            let dst = res.at_mut(out_row, res_limb);
            dst.fill(BE::ScalarBig::from(0));
        }

        let a_coeffs = a.at(a_col, a_limb);
        let pmat_raw = pmat.raw();
        let pmat_col_base = pmat_limb * pmat.cols_out();
        let pmat_nrows = pmat.rows() * pmat.cols_in();
        let pmat_ncols = pmat.cols_out() * pmat.size();
        let mut out_row = 0;

        if rows_out > 0 && !pmat_col_base.is_multiple_of(2) {
            let dst = res.at_mut(0, res_limb);
            coeff_mat1col_product(
                rows_in,
                pmat_nrows,
                pmat_ncols,
                pmat_col_base,
                &mut dst[0],
                a_coeffs,
                pmat_raw,
            );
            out_row = 1;
        }

        while out_row + 1 < rows_out {
            let pmat_col = pmat_col_base + out_row;
            let mut pair = [BE::ScalarBig::from(0); 2];
            coeff_mat2cols_product(rows_in, pmat_nrows, pmat_ncols, pmat_col, &mut pair, a_coeffs, pmat_raw);
            res.at_mut(out_row, res_limb)[0] = pair[0];
            res.at_mut(out_row + 1, res_limb)[0] = pair[1];
            out_row += 2;
        }

        if out_row < rows_out {
            let dst = res.at_mut(out_row, res_limb);
            coeff_mat1col_product(
                rows_in,
                pmat_nrows,
                pmat_ncols,
                pmat_col_base + out_row,
                &mut dst[0],
                a_coeffs,
                pmat_raw,
            );
        }
    }
}

impl<BE: Backend> CoeffMatPMatDefault<BE> for BE
where
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}
