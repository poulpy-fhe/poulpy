macro_rules! hal_impl_vmp_ntt126_ifma {
    () => {
        fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            let _ = (rows, cols_in, cols_out, size);
            crate::ntt126_ifma::vmp::vmp_prepare_tmp_bytes_ifma(module.n())
        }

        fn vmp_prepare<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
        where
            R: VmpPMatToMut<Self>,
            A: MatZnxToRef,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::vmp::vmp_prepare_tmp_bytes_ifma(module.n());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_prepare_ifma(module, res, a, tmp)
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
            let _ = (module, res_size, b_cols_out, b_size);
            crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
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
            let _ = (module, res_size, b_cols_out, b_size);
            crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            a: &A,
            b: &C,
            limb_offset: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            C: VmpPMatToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
            let b_ref: VmpPMat<&[u8], Self> = b.to_ref();
            let bytes = crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_ref.size(), b_ref.rows(), b_ref.cols_in());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_apply_dft_to_dft_ifma(module, res, a, b, limb_offset, tmp)
        }

        fn vmp_apply_dft_to_dft_accumulate<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            a: &A,
            b: &C,
            limb_offset: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            C: VmpPMatToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
            let b_ref: VmpPMat<&[u8], Self> = b.to_ref();
            let bytes = crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_ref.size(), b_ref.rows(), b_ref.cols_in());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_apply_dft_to_dft_accumulate_ifma(module, res, a, b, limb_offset, tmp)
        }

        fn vmp_zero<R>(_module: &Module<Self>, res: &mut R)
        where
            R: VmpPMatToMut<Self>,
        {
            crate::ntt126_ifma::vmp::vmp_zero(res)
        }
    };
}
