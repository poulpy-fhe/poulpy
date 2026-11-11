#[macro_export]
macro_rules! hal_impl_svp {
    ($defaults:ident, $ring:ty) => {
        fn svp_ppol_prepare(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::SvpPPolBackendMut<'_, Self, $ring>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self, $ring>,
            a_col: usize,
        ) {
            <Self as $defaults<$ring>>::svp_ppol_prepare_default(module, &mut res, res_col, a, a_col)
        }

        fn svp_ppol_copy(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpPPolBackendMut<'_, Self, $ring>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self, $ring>,
            a_col: usize,
        ) {
            <Self as $defaults<$ring>>::svp_ppol_copy_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self, $ring>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>,
            b_col: usize,
        ) {
            <Self as $defaults<$ring>>::svp_apply_dft_to_dft_default(module, res, res_col, &a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self, $ring>,
            a_col: usize,
        ) {
            <Self as $defaults<$ring>>::svp_apply_dft_to_dft_assign_default(module, res, res_col, &a, a_col)
        }
    };
}
