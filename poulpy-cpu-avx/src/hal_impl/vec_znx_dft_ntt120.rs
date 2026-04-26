macro_rules! hal_impl_vec_znx_dft_ntt120 {
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
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_apply_default(module, step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_idft_apply_tmp_bytes_default(module)
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
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_idft_apply_default(module, res, res_col, a, a_col, scratch)
        }

        fn vec_znx_idft_apply_tmpa<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxDftToMut<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_idft_apply_tmpa_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_add_into<R, A, D>(
            module: &Module<Self>,
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
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_add_into_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_add_scaled_assign<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            a_scale: i64,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_add_scaled_assign_default(
                module, res, res_col, a, a_col, a_scale,
            )
        }

        fn vec_znx_dft_add_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_add_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub<R, A, D>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            D: VecZnxDftToRef<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_sub_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_dft_sub_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_sub_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_sub_negate_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_sub_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_dft_copy<R, A>(
            module: &Module<Self>,
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
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_copy_default(module, step, offset, res, res_col, a, a_col)
        }

        fn vec_znx_dft_zero<R>(module: &Module<Self>, res: &mut R, res_col: usize)
        where
            R: VecZnxDftToMut<Self>,
        {
            <Self as NTT120VecZnxDftDefaults<Self>>::vec_znx_dft_zero_default(module, res, res_col)
        }

        fn vec_znx_idft_apply_consume<D: Data>(module: &Module<Self>, a: VecZnxDft<D, Self>) -> VecZnxBig<D, Self>
        where
            VecZnxDft<D, Self>: VecZnxDftToMut<Self>,
        {
            crate::ntt120::vec_znx_dft_consume::vec_znx_idft_apply_consume(module, a)
        }
    };
}
