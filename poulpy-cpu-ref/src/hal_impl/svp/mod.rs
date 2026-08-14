#[macro_use]
mod ppol;
#[macro_use]
mod tpol;

/// Emits the tier-independent SVP methods: the unprepared-operand variants,
/// all derived from the `tpol` tier.
#[macro_export]
macro_rules! hal_impl_svp {
    () => {
        $crate::__hal_impl_svp_derived!();
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hal_impl_svp_derived {
    () => {
        fn svp_apply_to_big_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize {
            module.bytes_of_vec_znx_dft(1, res_size)
        }

        fn svp_apply_to_small_tmp_bytes(module: &Module<Self>, b_size: usize) -> usize {
            module.bytes_of_vec_znx_dft(1, b_size)
                + module.bytes_of_vec_znx_big(1, b_size)
                + module.vec_znx_big_normalize_tmp_bytes()
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
            let mut tpol = poulpy_hal::layouts::SvpTPolOwned::<Self>::alloc(module.n(), 1);
            Self::svp_prepare_tpol(module, &mut tpol.to_backend_mut(), 0, a, a_col);
            Self::svp_apply_tpol_small_to_dft(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
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
            let mut tpol = poulpy_hal::layouts::SvpTPolOwned::<Self>::alloc(module.n(), 1);
            Self::svp_prepare_tpol(module, &mut tpol.to_backend_mut(), 0, a, a_col);
            Self::svp_apply_tpol_dft_to_dft(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
        }

        fn svp_apply_small_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            let mut tpol = poulpy_hal::layouts::SvpTPolOwned::<Self>::alloc(module.n(), 1);
            Self::svp_prepare_tpol(module, &mut tpol.to_backend_mut(), 0, a, a_col);
            Self::svp_apply_tpol_dft_to_dft_assign(module, res, res_col, &tpol.to_backend_ref(), 0);
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
            let res_size: usize = res.size();
            let (mut tmp, _) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
            Self::svp_apply_small_small_to_dft(module, &mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
            module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
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
            let b_size: usize = b.size();
            let (mut tmp_dft, scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
            Self::svp_apply_small_small_to_dft(module, &mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
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
            let res_size: usize = res.size();
            let (mut tmp, _) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
            Self::svp_apply_small_dft_to_dft(module, &mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
            module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
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
            let b_size: usize = b.size();
            let (mut tmp_dft, scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
            Self::svp_apply_small_dft_to_dft(module, &mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
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
    };
}
