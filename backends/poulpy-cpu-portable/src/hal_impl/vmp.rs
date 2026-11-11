#[macro_export]
macro_rules! hal_impl_vmp {
    ($defaults:ident, $ring:ty) => {
        fn vmp_pmat_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as $defaults<$ring>>::vmp_pmat_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn vmp_pmat_prepare(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self, $ring>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self, $ring>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<$ring>>::vmp_pmat_prepare_default(module, res, a, &mut scratch);
        }

        fn vmp_apply_dft_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<$ring>>::vmp_apply_dft_to_dft_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
        }

        fn vmp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self, $ring>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<$ring>>::vmp_apply_dft_to_dft_default(module, res, a, b, limb_offset, &mut scratch);
        }

        fn vmp_apply_dft_to_dft_add_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<$ring>>::vmp_apply_dft_to_dft_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
        }

        fn vmp_apply_dft_to_dft_add(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self, $ring>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<$ring>>::vmp_apply_dft_to_dft_add_default(module, res, a, b, limb_offset, &mut scratch);
        }

        fn vmp_extract_selected_rows(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self, $ring>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self, $ring>,
            first_row: usize,
            row_step: usize,
        ) {
            <Self as $defaults<$ring>>::vmp_extract_selected_rows_default(module, res, a, first_row, row_step)
        }

        fn vmp_zero(module: &Module<Self>, res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self, $ring>) {
            <Self as $defaults<$ring>>::vmp_zero_default(module, res)
        }
    };
}
