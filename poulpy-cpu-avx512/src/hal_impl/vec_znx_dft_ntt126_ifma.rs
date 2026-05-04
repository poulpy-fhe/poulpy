macro_rules! hal_impl_vec_znx_dft_ntt126_ifma {
    () => {
        fn vec_znx_dft_apply<R, A>(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxToRef,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
        }

        fn vec_znx_idft_apply<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let (tmp, _) = scratch.take_slice::<u64>(
                crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n()) / std::mem::size_of::<u64>(),
            );
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp)
        }

        fn vec_znx_idft_apply_tmpa<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxDftToMut<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmpa(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_add_into<R, A, D>(
            _module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &D,
            b_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            D: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_add_scaled_assign<R, A>(
            _module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            a_scale: i64,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale)
        }

        fn vec_znx_dft_add_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_assign(res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub<R, A, D>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            D: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_sub_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub_assign(res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub_negate_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub_negate_assign(res, res_col, a, a_col)
        }

        fn vec_znx_dft_copy<R, A>(
            _module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_copy(step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_dft_zero<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
        where
            R: VecZnxDftToMut<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_zero(res, res_col)
        }

        fn vec_znx_idft_apply_consume<D: Data>(module: &Module<Self>, a: VecZnxDft<D, Self>) -> VecZnxBig<D, Self>
        where
            VecZnxDft<D, Self>: VecZnxDftToMut<Self>,
        {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_consume(module, a)
        }
    };
}
