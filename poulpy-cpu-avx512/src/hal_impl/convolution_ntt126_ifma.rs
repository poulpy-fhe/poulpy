macro_rules! hal_impl_convolution_ntt126_ifma {
    () => {
        fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_left_tmp_bytes(module.n())
        }

        fn cnv_prepare_left<R, A>(module: &Module<Self>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<Self>)
        where
            R: CnvPVecLToMut<Self>,
            A: VecZnxToRef,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_left_tmp_bytes(module.n());
            let (tmp, _) = scratch.take_slice::<u8>(bytes);
            crate::ntt126_ifma::convolution::cnv_prepare_left(module, res, a, mask, tmp)
        }

        fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_right_tmp_bytes(module.n())
        }

        fn cnv_prepare_right<R, A>(module: &Module<Self>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<Self>)
        where
            R: CnvPVecRToMut<Self>,
            A: VecZnxToRef + ZnxInfos,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_right_tmp_bytes(module.n());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt126_ifma::convolution::cnv_prepare_right(module, res, a, mask, tmp)
        }

        fn cnv_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            _res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(a_size, b_size)
        }

        fn cnv_by_const_apply_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply<R, A>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &[i64],
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_by_const_apply_tmp_bytes(0, 0, 0);
            let (tmp, _) = scratch.take_slice::<u8>(bytes);
            crate::ntt126_ifma::convolution::cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, tmp)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft<R, A, B>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &B,
            b_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: CnvPVecLToRef<Self>,
            B: CnvPVecRToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(a.to_ref().size(), b.to_ref().size());
            let (tmp, _) = scratch.take_slice::<u8>(bytes);
            unsafe { crate::ntt126_ifma::convolution::cnv_apply_dft_ifma(res, cnv_offset, res_col, a, a_col, b, b_col, tmp) }
        }

        fn cnv_pairwise_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_dft<R, A, B>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            b: &B,
            i: usize,
            j: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: CnvPVecLToRef<Self>,
            B: CnvPVecRToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(
                res.to_mut().size(),
                a.to_ref().size(),
                b.to_ref().size(),
            );
            let (tmp, _) = scratch.take_slice::<u8>(bytes);
            unsafe { crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma(res, cnv_offset, res_col, a, b, i, j, tmp) }
        }

        fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_self_tmp_bytes(module.n())
        }

        fn cnv_prepare_self<L, R, A>(
            module: &Module<Self>,
            left: &mut L,
            right: &mut R,
            a: &A,
            mask: i64,
            scratch: &mut Scratch<Self>,
        ) where
            L: CnvPVecLToMut<Self>,
            R: CnvPVecRToMut<Self>,
            A: VecZnxToRef + ZnxInfos,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_self_tmp_bytes(module.n());
            let (tmp, _) = scratch.take_slice::<u8>(bytes);
            crate::ntt126_ifma::convolution::cnv_prepare_self(module, left, right, a, mask, tmp)
        }
    };
}
