macro_rules! hal_impl_svp_ntt126_ifma {
    () => {
        fn svp_prepare<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: SvpPPolToMut<Self>,
            A: ScalarZnxToRef,
        {
            crate::ntt126_ifma::svp::svp_prepare(module, res, res_col, a, a_col)
        }

        fn svp_apply_dft<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: SvpPPolToRef<Self>,
            C: VecZnxToRef,
        {
            crate::ntt126_ifma::svp::svp_apply_dft(module, res, res_col, a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            A: SvpPPolToRef<Self>,
            C: VecZnxDftToRef<Self>,
        {
            crate::ntt126_ifma::svp::svp_apply_dft_to_dft(module, res, res_col, a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: SvpPPolToRef<Self>,
        {
            crate::ntt126_ifma::svp::svp_apply_dft_to_dft_assign(module, res, res_col, a, a_col)
        }
    };
}
