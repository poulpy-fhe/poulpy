//! Backend extension points for scalar-vector product (SVP) operations
//! on [`SvpPPol`](poulpy_hal::layouts::SvpPPol).

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        svp::{
            svp_apply_dft as fft64_svp_apply_dft, svp_apply_dft_to_dft as fft64_svp_apply_dft_to_dft,
            svp_apply_dft_to_dft_assign as fft64_svp_apply_dft_to_dft_assign, svp_prepare as fft64_svp_prepare,
        },
    },
    ntt120::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero,
        ntt::NttTable,
        primes::Primes30,
        svp::{ntt120_svp_apply_dft_to_dft, ntt120_svp_apply_dft_to_dft_assign, ntt120_svp_prepare},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::VecZnxDftApply,
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos,
    },
};

#[doc(hidden)]
pub trait FFT64SvpDefaults<BE: Backend>: Backend {
    fn svp_prepare_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        R: SvpPPolToMut<BE>,
        A: ScalarZnxToRef,
    {
        fft64_svp_prepare::<R, A, BE>(module.get_fft_table(), res, res_col, a, a_col);
    }

    fn svp_apply_dft_default<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
        C: VecZnxToRef,
    {
        fft64_svp_apply_dft::<R, A, C, BE>(module.get_fft_table(), res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
        C: VecZnxDftToRef<BE>,
    {
        fft64_svp_apply_dft_to_dft::<R, A, C, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        fft64_svp_apply_dft_to_dft_assign::<R, A, BE>(res, res_col, a, a_col);
    }
}

impl<BE: Backend> FFT64SvpDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120SvpDefaults<BE: Backend>: Backend {
    fn svp_prepare_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        R: SvpPPolToMut<BE>,
        A: ScalarZnxToRef,
    {
        ntt120_svp_prepare::<R, A, BE>(module, res, res_col, a, a_col);
    }

    fn svp_apply_dft_default<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttMulBbc + NttZero,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
        C: VecZnxToRef,
    {
        let b = b.to_ref();
        let b_size = b.size();
        let mut b_dft = poulpy_hal::layouts::VecZnxDftOwned::<BE>::alloc(module.n(), 1, b_size);

        <Module<BE> as VecZnxDftApply<BE>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, &b, b_col);
        ntt120_svp_apply_dft_to_dft::<R, A, _, BE>(module, res, res_col, a, a_col, &b_dft, 0);
    }

    fn svp_apply_dft_to_dft_default<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc + NttZero,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
        C: VecZnxDftToRef<BE>,
    {
        ntt120_svp_apply_dft_to_dft::<R, A, C, BE>(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        ntt120_svp_apply_dft_to_dft_assign::<R, A, BE>(module, res, res_col, a, a_col);
    }
}

impl<BE: Backend> NTT120SvpDefaults<BE> for BE {}
