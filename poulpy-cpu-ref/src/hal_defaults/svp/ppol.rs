//! Reference kernels whose prepared operand is the packed cold-prep [`SvpPPol`](poulpy_hal::layouts::SvpPPol).

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        svp as fft64_svp,
    },
    ntt4x30::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero, ntt::NttTable, primes::Primes30, svp as ntt4x30_svp,
        types::Q120bScalar, vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::VecZnxDftApply,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxView, ZnxViewMut,
    },
};

#[doc(hidden)]
pub trait FFT64SvpPPolDefault<BE: Backend<ZnxWord = i64>>: Backend {
    fn svp_prepare_ppol_default(
        module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_prepare_ppol::<BE>(module.get_fft_table(), res, res_col, a, a_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_ppol_small_to_dft::<BE>(module.get_fft_table(), res, res_col, a, a_col, b, b_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_dft_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_ppol_dft_to_dft::<BE>(res, res_col, a, a_col, b, b_col);
    }
    fn svp_apply_ppol_dft_to_dft_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_ppol_dft_to_dft_assign::<BE>(res, res_col, a, a_col);
    }
    fn svp_ppol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64SvpPPolDefault<BE> for BE where BE::OwnedBuf: HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30SvpPPolDefault<BE: Backend<ZnxWord = i64>>: Backend {
    fn svp_prepare_ppol_default(
        module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_prepare_ppol::<BE>(module, res, res_col, a, a_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut b_dft = poulpy_hal::layouts::VecZnxDftOwned::<BE>::alloc(module.n(), 1, b.size());
        module.vec_znx_dft_apply(1, 0, &mut b_dft.to_backend_mut(), 0, b, b_col);
        ntt4x30_svp::ntt4x30_svp_apply_ppol_dft_to_dft::<BE>(module, res, res_col, a, a_col, &b_dft.to_backend_ref(), 0);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_apply_ppol_dft_to_dft::<BE>(module, res, res_col, a, a_col, b, b_col);
    }
    fn svp_apply_ppol_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_apply_ppol_dft_to_dft_assign::<BE>(module, res, res_col, a, a_col);
    }
    fn svp_ppol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30SvpPPolDefault<BE> for BE where BE::OwnedBuf: HostDataMut {}
