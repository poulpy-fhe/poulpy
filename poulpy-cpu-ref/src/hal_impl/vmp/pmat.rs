//! `pmat` tier of the VMP backend implementation.
//!
//! Expanded inside a backend's `HalVmpPmatImpl` block.

/// Emits every `pmat`-tier method: the kernel forwarders, then the variants
/// derived from them.
#[macro_export]
macro_rules! hal_impl_vmp_pmat {
    ($defaults:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_dft_accumulate(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $defaults<Self>>::vmp_apply_pmat_dft_to_dft_accumulate_default(module, res, a, b, limb_offset, scratch)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes_default(
                module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $defaults<Self>>::vmp_apply_pmat_dft_to_dft_default(module, res, a, b, limb_offset, scratch)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::vmp_apply_pmat_dft_to_dft_tmp_bytes_default(
                module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_prepare_pmat(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <Self as $defaults<Self>>::vmp_prepare_pmat_default(module, res, a, scratch)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_prepare_pmat_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as $defaults<Self>>::vmp_prepare_pmat_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }
        $crate::__hal_impl_vmp_pmat_derived!();
    };

    // `kernels: skip` emits only the derived variants, for a backend that
    // writes its own kernels in the same impl block.
    (kernels: skip) => {
        $crate::__hal_impl_vmp_pmat_derived!();
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hal_impl_vmp_pmat_derived {
    () => {
        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b_base2k: usize,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let cols_out = res.cols();
            let res_size = res.size();
            let (mut tmp, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, cols_out, res_size);
            Self::vmp_apply_pmat_dft_to_big(module, &mut tmp.to_backend_mut(), a, b, limb_offset, &mut scratch_1);
            let tmp_ref = tmp.to_backend_ref();
            for col in 0..cols_out {
                module.vec_znx_big_normalize(
                    res,
                    res_base2k,
                    res_offset,
                    col,
                    &tmp_ref,
                    b_base2k,
                    col,
                    &mut scratch_1,
                );
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_small_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vec_znx_big(a_cols_out, res_size)
                + module.vec_znx_big_normalize_tmp_bytes()
                + Self::vmp_apply_pmat_dft_to_big_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_small(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_offset: i64,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_base2k: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let cols_out = res.cols();
            let res_size = res.size();
            let (mut tmp, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, cols_out, res_size);
            Self::vmp_apply_pmat_small_to_big(module, &mut tmp.to_backend_mut(), a, b, &mut scratch_1);
            let tmp_ref = tmp.to_backend_ref();
            for col in 0..cols_out {
                module.vec_znx_big_normalize(
                    res,
                    res_base2k,
                    res_offset,
                    col,
                    &tmp_ref,
                    b_base2k,
                    col,
                    &mut scratch_1,
                );
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_small_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vec_znx_big(a_cols_out, res_size)
                + module.vec_znx_big_normalize_tmp_bytes()
                + Self::vmp_apply_pmat_small_to_big_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let cols_out = res.cols();
            let res_size = res.size();
            let (mut tmp, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, cols_out, res_size);
            Self::vmp_apply_pmat_dft_to_dft(module, &mut tmp.to_backend_mut(), a, b, limb_offset, &mut scratch_1);
            for col in 0..cols_out {
                module.vec_znx_idft_apply_tmpa(res, col, &mut tmp.to_backend_mut(), col);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_dft_to_big_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vec_znx_dft(a_cols_out, res_size)
                + Self::vmp_apply_pmat_dft_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_big(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let cols_out = res.cols();
            let res_size = res.size();
            let (mut tmp, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, cols_out, res_size);
            Self::vmp_apply_pmat_small_to_dft(module, &mut tmp.to_backend_mut(), a, b, &mut scratch_1);
            for col in 0..cols_out {
                module.vec_znx_idft_apply_tmpa(res, col, &mut tmp.to_backend_mut(), col);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_big_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vec_znx_dft(a_cols_out, res_size)
                + Self::vmp_apply_pmat_small_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_dft_accumulate(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let cols_out = res.cols();
            let res_size = res.size();
            let (mut tmp, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, cols_out, res_size);
            for col in 0..cols_out {
                module.vec_znx_dft_zero(&mut tmp.to_backend_mut(), col);
            }
            Self::vmp_apply_pmat_small_to_dft(module, &mut tmp.to_backend_mut(), a, b, &mut scratch_1);
            let tmp_ref = tmp.to_backend_ref();
            for col in 0..cols_out {
                module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            module.bytes_of_vec_znx_dft(a_cols_out, res_size)
                + Self::vmp_apply_pmat_small_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
        }

        /// Lifts the vector into a scratch `VecZnxDft`, right-aligning its columns
        /// against the matrix's `cols_in`, then runs the DFT-domain kernel. No
        /// backend has a fused coefficient-input kernel, so this is shared.
        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_dft(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let dft_size = b.size().min(a.rows());
            let (mut b_dft, mut scratch_1) =
                poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, a.cols_in(), dft_size);
            let cols_to_copy = b.cols().min(a.cols_in());
            let b_start_col = b.cols() - cols_to_copy;
            let offset = a.cols_in() - cols_to_copy;
            for j in 0..offset {
                <Module<Self> as poulpy_hal::api::VecZnxDftZero<Self>>::vec_znx_dft_zero(module, &mut b_dft.to_backend_mut(), j);
            }
            for j in 0..cols_to_copy {
                <Module<Self> as poulpy_hal::api::VecZnxDftApply<Self>>::vec_znx_dft_apply(
                    module,
                    1,
                    0,
                    &mut b_dft.to_backend_mut(),
                    offset + j,
                    b,
                    b_start_col + j,
                );
            }
            Self::vmp_apply_pmat_dft_to_dft(module, res, a, &b_dft.to_backend_ref(), 0, &mut scratch_1);
        }

        #[allow(clippy::too_many_arguments)]
        fn vmp_apply_pmat_small_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_rows: usize,
            a_cols_in: usize,
            a_cols_out: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            let dft_size = b_size.min(a_rows);
            module.bytes_of_vec_znx_dft(a_cols_in, dft_size)
                + Self::vmp_apply_pmat_dft_to_dft_tmp_bytes(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, dft_size)
        }
    };
}
