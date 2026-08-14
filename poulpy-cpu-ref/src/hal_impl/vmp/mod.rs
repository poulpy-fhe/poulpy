#[macro_use]
mod pmat;
#[macro_use]
mod tmat;

/// Emits the tier-independent VMP methods: the unprepared-operand variants,
/// all derived from the `tmat` tier.
#[macro_export]
macro_rules! hal_impl_vmp {
    () => {
        $crate::__hal_impl_vmp_derived!();
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hal_impl_vmp_derived {
    () => {
        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_small_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_small_to_dft(module, res, &tmat.to_backend_ref(), b, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_dft_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_dft_to_dft(module, res, &tmat.to_backend_ref(), b, limb_offset, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(
                        module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size,
                    ),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_dft_accumulate(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_small_to_dft_accumulate(module, res, &tmat.to_backend_ref(), b, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(
                        module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size,
                    ),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_dft_accumulate(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_dft_to_dft_accumulate(module, res, &tmat.to_backend_ref(), b, limb_offset, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_big_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_small_to_big_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_small_to_big(module, res, &tmat.to_backend_ref(), b, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_big_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_dft_to_big_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_dft_to_big(module, res, &tmat.to_backend_ref(), b, limb_offset, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_small_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_small_to_small_tmp_bytes(
                        module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size,
                    ),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_small_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_base2k: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_small_to_small(
                module,
                res,
                res_base2k,
                res_offset,
                &tmat.to_backend_ref(),
                b,
                b_base2k,
                &mut scratch_1,
            );
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_small_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vmp_tmat(a_rows, a_cols_in, a_cols_out, a_size)
                + Self::vmp_prepare_tmat_tmp_bytes(module, a_rows, a_cols_in, a_cols_out, a_size).max(
                    Self::vmp_apply_tmat_dft_to_small_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size),
                )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_small_dft_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_base2k: usize,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let (mut tmat, mut scratch_1) = poulpy_hal::api::ScratchArenaTakeBasic::take_vmp_tmat_scratch(
                scratch.borrow(),
                module,
                a.rows(),
                a.cols_in(),
                a.cols_out(),
                a.size(),
            );
            Self::vmp_prepare_tmat(module, &mut tmat.to_backend_mut(), a, &mut scratch_1);
            Self::vmp_apply_tmat_dft_to_small(
                module,
                res,
                res_base2k,
                res_offset,
                &tmat.to_backend_ref(),
                b,
                b_base2k,
                limb_offset,
                &mut scratch_1,
            );
        }
    };
}
