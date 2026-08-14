//! `tpol` tier of the SVP backend implementation.
//!
//! Expanded inside a backend's `HalSvpTpolImpl` block.

/// Emits every `tpol`-tier method: the kernel forwarders, then the variants
/// derived from them.
#[macro_export]
macro_rules! hal_impl_svp_tpol {
    ($defaults:ident) => {
        fn svp_apply_tpol_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_apply_tpol_dft_to_dft_assign_default(module, res, res_col, a, a_col)
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

        fn svp_tpol_copy_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpTPolBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::svp_tpol_copy_backend_default(module, res, res_col, a, a_col)
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

        $crate::__hal_impl_svp_tpol_derived!();
    };

    // `kernels: skip` emits only the derived variants, for a backend that
    // writes its own kernels in the same impl block.
    (kernels: skip) => {
        $crate::__hal_impl_svp_tpol_derived!();
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hal_impl_svp_tpol_derived {
    () => {
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
            let b_size: usize = b.size();
            let (mut tmp_dft, scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
            Self::svp_apply_tpol_dft_to_dft(module, &mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
            let (mut tmp_big, mut scratch_2) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch_1, module, 1, b_size);
            module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
            module.vec_znx_big_normalize(
                res,
                res_base2k,
                res_offset,
                res_col,
                &tmp_big.to_backend_ref(),
                b_base2k,
                0,
                &mut scratch_2,
            );
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
            let res_size: usize = res.size();
            let (mut tmp, _) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
            Self::svp_apply_tpol_dft_to_dft(module, &mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
            module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
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
            let b_size: usize = b.size();
            let (mut tmp_dft, scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
            Self::svp_apply_tpol_small_to_dft(module, &mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
            let (mut tmp_big, mut scratch_2) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch_1, module, 1, b_size);
            module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
            module.vec_znx_big_normalize(
                res,
                res_base2k,
                res_offset,
                res_col,
                &tmp_big.to_backend_ref(),
                b_base2k,
                0,
                &mut scratch_2,
            );
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
            let res_size: usize = res.size();
            let (mut tmp, _) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
            Self::svp_apply_tpol_small_to_dft(module, &mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
            module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
        }
    };
}
