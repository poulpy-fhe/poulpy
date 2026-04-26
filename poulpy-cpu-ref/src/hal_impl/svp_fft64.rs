macro_rules! hal_impl_svp_fft64 {
    () => {
        fn svp_prepare<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: SvpPPolToMut<Self>,
            A: ScalarZnxToRef,
        {
            <Self as FFT64SvpDefaults<Self>>::svp_prepare_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_dft<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: SvpPPolToRef<Self>,
            C: VecZnxToRef,
        {
            <Self as FFT64SvpDefaults<Self>>::svp_apply_dft_default(module, res, res_col, a, a_col, b, b_col)
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
            <Self as FFT64SvpDefaults<Self>>::svp_apply_dft_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxDftToMut<Self>,
            A: SvpPPolToRef<Self>,
        {
            <Self as FFT64SvpDefaults<Self>>::svp_apply_dft_to_dft_assign_default(module, res, res_col, a, a_col)
        }
    };
}
