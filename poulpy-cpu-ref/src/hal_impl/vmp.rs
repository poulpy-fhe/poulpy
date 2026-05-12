#[macro_export]
macro_rules! hal_impl_vmp {
    ($defaults:ident) => {
        fn vmp_apply_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            let a_dft_size = a_size.min(b_rows);
            <Self as Backend>::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
                + Self::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size)
        }

        fn vmp_apply_dft<'s, R>(
            module: &Module<Self>,
            res: &mut R,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxDftToBackendMut<Self>,
        {
            let a_cols = <poulpy_hal::layouts::VecZnxBackendRef<'_, Self> as ZnxInfos>::cols(a);
            let a_size = <poulpy_hal::layouts::VecZnxBackendRef<'_, Self> as ZnxInfos>::size(a);
            let b_rows = <poulpy_hal::layouts::VmpPMatBackendRef<'_, Self> as ZnxInfos>::rows(b);
            let cols_to_copy = a_cols.min(b.cols_in());
            let a_start_col = a_cols - cols_to_copy;
            let a_dft_size = a_size.min(b_rows);
            let offset = b.cols_in() - cols_to_copy;

            scratch.consume(|scratch| {
                let (mut a_dft, mut scratch) =
                    poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch, module, b.cols_in(), a_dft_size);

                for j in 0..offset {
                    module.vec_znx_dft_zero(&mut a_dft, j);
                }

                for j in 0..cols_to_copy {
                    module.vec_znx_dft_apply(1, 0, &mut a_dft, offset + j, a, a_start_col + j);
                }

                let mut res_ref = res.to_backend_mut();
                module.vmp_apply_dft_to_dft(&mut res_ref, &a_dft.to_backend_ref(), b, 0, &mut scratch);
                ((), scratch)
            })
        }

        fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as $defaults<Self>>::vmp_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn vmp_prepare<'s>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::MatZnxBackendRef<'_, Self>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
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

        fn vmp_apply_dft_to_dft<'s, 'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, Self>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_default(module, res, a, b, limb_offset, &mut scratch);
        }

        fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
        }

        fn vmp_apply_dft_to_dft_accumulate<'s, 'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, Self>,
            a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vmp_apply_dft_to_dft_accumulate_default(module, res, a, b, limb_offset, &mut scratch);
        }

        fn vmp_zero(module: &Module<Self>, res: &mut poulpy_hal::layouts::VmpPMatBackendMut<'_, Self>) {
            <Self as $defaults<Self>>::vmp_zero_default(module, res)
        }
    };
}
