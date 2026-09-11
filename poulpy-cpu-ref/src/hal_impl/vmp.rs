#[macro_export]
macro_rules! hal_impl_vmp {
    ($defaults:ident) => {
        fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as $defaults<Self>>::vmp_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn vmp_prepare(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vmp_prepare_default(module, res, a, &mut scratch);
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
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
        }

        fn vmp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_default(module, res, a, b, limb_offset, &mut scratch);
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
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_add_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
        }

        fn vmp_apply_dft_to_dft_add(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_add_default(module, res, a, b, limb_offset, &mut scratch);
        }

        fn vmp_extract_selected_rows(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            first_row: usize,
            row_step: usize,
        ) {
            <Self as $defaults<Self>>::vmp_extract_selected_rows_default(module, res, a, first_row, row_step)
        }

        fn vmp_zero(module: &Module<Self>, res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>) {
            <Self as $defaults<Self>>::vmp_zero_default(module, res)
        }
    };
}
