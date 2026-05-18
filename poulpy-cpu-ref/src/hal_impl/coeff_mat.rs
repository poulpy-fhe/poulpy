#[macro_export]
macro_rules! hal_impl_coeff_mat {
    ($defaults:ident) => {
        fn coeff_mat_prepare_tmp_bytes(
            module: &Module<Self>,
            rows: usize,
            cols_in: usize,
            cols_out: usize,
            size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::coeff_mat_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn coeff_mat_prepare(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CoeffMatPMatBackendMut<'_, Self>,
            matrix: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $defaults<Self>>::coeff_mat_prepare_default(module, res, matrix, scratch)
        }

        fn coeff_mat_apply_big_tmp_bytes(module: &Module<Self>, rows_in: usize, rows_out: usize) -> usize {
            <Self as $defaults<Self>>::coeff_mat_apply_big_tmp_bytes_default(module, rows_in, rows_out)
        }

        fn coeff_mat_apply_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_limb: usize,
            pmat: &poulpy_hal::layouts::CoeffMatPMatBackendRef<'_, Self>,
            pmat_limb: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            a_limb: usize,
            rows_in: usize,
            rows_out: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $defaults<Self>>::coeff_mat_apply_big_default(
                module, res, res_limb, pmat, pmat_limb, a, a_col, a_limb, rows_in, rows_out, scratch,
            )
        }
    };
}
