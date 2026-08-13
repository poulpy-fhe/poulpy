#[macro_export]
macro_rules! hal_impl_svp {
    ($defaults:ident) => {
        fn svp_prepare_ppol(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_prepare_ppol_default(module, res, res_col, a, a_col)
        }

        fn svp_prepare_tpol(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpTPolBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_prepare_tpol_default(module, res, res_col, a, a_col)
        }

        fn svp_ppol_copy_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_ppol_copy_backend_default(module, res, res_col, a, a_col)
        }

        fn svp_tpol_copy_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpTPolBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_tpol_copy_backend_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_to_big_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_to_big_tmp_bytes_default(module, res_size)
        }

        fn svp_apply_to_small_tmp_bytes(module: &Module<Self>, b_size: usize) -> usize {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_to_small_tmp_bytes_default(module, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_small_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_small_small_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_small_dft_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_small_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_tpol_small_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_tpol_dft_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_small_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_ppol_small_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_ppol_dft_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_small_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_small_small_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_dft_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_small_dft_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_small_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_tpol_small_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_dft_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_tpol_dft_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_small_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_ppol_small_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_dft_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_ppol_dft_to_big_default(
                module, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_small_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_small_small_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_small_dft_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_small_dft_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_small_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_tpol_small_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_tpol_dft_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_tpol_dft_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_small_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_ppol_small_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_ppol_dft_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_base2k: usize,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $crate::hal_defaults::SvpDerivedDefault<Self>>::svp_apply_ppol_dft_to_small_default(
                module, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
            )
        }

        fn svp_apply_small_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_small_dft_to_dft_assign_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_tpol_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_tpol_dft_to_dft_assign_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_ppol_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_ppol_dft_to_dft_assign_default(module, res, res_col, a, a_col)
        }
    };
}
