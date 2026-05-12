#[macro_export]
macro_rules! hal_impl_vec_znx_dft {
    ($defaults:ident) => {
        fn vec_znx_dft_apply(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_apply_default(module, step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as $defaults<Self>>::vec_znx_idft_apply_tmp_bytes_default(module)
        }

        fn vec_znx_idft_apply<'s>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vec_znx_idft_apply_default(module, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_idft_apply_tmpa(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_idft_apply_tmpa_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_add_into(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_add_into_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_add_scaled_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            a_scale: i64,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_add_scaled_assign_default(module, res, res_col, a, a_col, a_scale)
        }

        fn vec_znx_dft_add_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_add_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_sub_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_sub_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_sub_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub_negate_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_sub_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_copy(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_dft_copy_default(module, step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_dft_zero(module: &Module<Self>, res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>, res_col: usize) {
            <Self as $defaults<Self>>::vec_znx_dft_zero_default(module, res, res_col)
        }
    };
}
