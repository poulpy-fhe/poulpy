//! Backend extension points for scalar-vector product (SVP) operations.
//!
//! Each flavor trait carries one method per (tier, shape) pair, each taking its
//! prepared operand as the concrete layout type and calling the matching
//! reference kernel. On top of those sit the three `small`-scalar variants,
//! which prepare into a temporary [`SvpTPol`](poulpy_hal::layouts::SvpTPol) and
//! then run the corresponding `tpol` method.
//!
//! `SvpTPol` and `SvpPPol` currently hold the same bytes on every CPU backend,
//! so the paired `tpol` and `ppol` methods do the same work. They stay separate
//! methods over separate types: a backend that gains a cheaper hot-prep form
//! repoints its `tpol` methods alone, and no caller changes.
//!
//! The derived `_to_big` and `_to_small` outputs live in [`SvpDerivedDefault`],
//! which is flavor-agnostic.

use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, SvpApplyPPolDftToDft, SvpApplyPPolSmallToDft, SvpApplySmallDftToDft, SvpApplySmallSmallToDft,
        SvpApplyTPolDftToDft, SvpApplyTPolSmallToDft, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef,
        SvpTPolBackendMut, SvpTPolBackendRef, SvpTPolOwned, SvpTPolToBackendMut, SvpTPolToBackendRef, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxBigBackendMut, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxView, ZnxViewMut,
    },
};

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

#[doc(hidden)]
pub trait FFT64SvpDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: HostDataMut,
{
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
    fn svp_prepare_tpol_default(
        module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_prepare_tpol::<BE>(module.get_fft_table(), res, res_col, a, a_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_tpol_small_to_dft::<BE>(module.get_fft_table(), res, res_col, a, a_col, b, b_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_dft_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_tpol_dft_to_dft::<BE>(res, res_col, a, a_col, b, b_col);
    }
    fn svp_apply_tpol_dft_to_dft_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        fft64_svp::svp_apply_tpol_dft_to_dft_assign::<BE>(res, res_col, a, a_col);
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

    fn svp_tpol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_small_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    fn svp_apply_small_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_assign_default(module, res, res_col, &tpol.to_backend_ref(), 0);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64SvpDefault<BE> for BE where BE::OwnedBuf: HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30SvpDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: HostDataMut,
{
    fn svp_prepare_ppol_default(
        module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
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
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
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
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
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
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_apply_ppol_dft_to_dft_assign::<BE>(module, res, res_col, a, a_col);
    }
    fn svp_prepare_tpol_default(
        module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_prepare_tpol::<BE>(module, res, res_col, a, a_col);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut b_dft = poulpy_hal::layouts::VecZnxDftOwned::<BE>::alloc(module.n(), 1, b.size());
        module.vec_znx_dft_apply(1, 0, &mut b_dft.to_backend_mut(), 0, b, b_col);
        ntt4x30_svp::ntt4x30_svp_apply_tpol_dft_to_dft::<BE>(module, res, res_col, a, a_col, &b_dft.to_backend_ref(), 0);
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_apply_tpol_dft_to_dft::<BE>(module, res, res_col, a, a_col, b, b_col);
    }
    fn svp_apply_tpol_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        ntt4x30_svp::ntt4x30_svp_apply_tpol_dft_to_dft_assign::<BE>(module, res, res_col, a, a_col);
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

    fn svp_tpol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_small_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    fn svp_apply_small_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30>>
            + NttFromZnx64
            + NttCFromB
            + NttMulBbc
            + NttZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_assign_default(module, res, res_col, &tpol.to_backend_ref(), 0);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30SvpDefault<BE> for BE where BE::OwnedBuf: HostDataMut {}

/// Derived SVP outputs, built on the `_to_dft` family plus IDFT and normalize.
///
/// Flavor-agnostic: every backend gets these for free once it implements the
/// `_to_dft` family.
#[doc(hidden)]
pub trait SvpDerivedDefault<BE: Backend<ZnxWord = i64>>: Backend {
    /// Scratch for a `_to_big` apply: one `res_size`-limb DFT intermediate.
    fn svp_apply_to_big_tmp_bytes_default(module: &Module<BE>, res_size: usize) -> usize
    where
        Module<BE>: VecZnxDftBytesOf,
    {
        module.bytes_of_vec_znx_dft(1, res_size)
    }

    /// Scratch for a `_to_small` apply: the DFT and big intermediates are
    /// carved at the input limb count so the product keeps full width until
    /// the normalization, which follows with its own scratch.
    fn svp_apply_to_small_tmp_bytes_default(module: &Module<BE>, b_size: usize) -> usize
    where
        Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    {
        module.bytes_of_vec_znx_dft(1, b_size) + module.bytes_of_vec_znx_big(1, b_size) + module.vec_znx_big_normalize_tmp_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplySmallSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_small_small_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplySmallSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_small_small_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplySmallDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_small_dft_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplySmallDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_small_dft_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplyTPolSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_tpol_small_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplyTPolSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_tpol_small_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplyTPolDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_tpol_dft_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplyTPolDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_tpol_dft_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplyPPolSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_ppol_small_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplyPPolSmallToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_ppol_small_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_big_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: SvpApplyPPolDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxDftBytesOf,
    {
        let res_size: usize = res.size();
        let (mut tmp, _) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_size);
        module.svp_apply_ppol_dft_to_dft(&mut tmp.to_backend_mut(), 0, a, a_col, b, b_col);
        module.vec_znx_idft_apply_tmpa(res, res_col, &mut tmp.to_backend_mut(), 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_small_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>:
            SvpApplyPPolDftToDft<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf,
    {
        let b_size: usize = b.size();
        let (mut tmp_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, b_size);
        module.svp_apply_ppol_dft_to_dft(&mut tmp_dft.to_backend_mut(), 0, a, a_col, b, b_col);
        let (mut tmp_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, 1, b_size);
        module.vec_znx_idft_apply_tmpa(&mut tmp_big.to_backend_mut(), 0, &mut tmp_dft.to_backend_mut(), 0);
        module.vec_znx_big_normalize(
            res,
            res_base2k,
            res_offset,
            res_col,
            &tmp_big.to_backend_ref(),
            b_base2k,
            0,
            &mut scratch_2,
        );
    }
}

impl<BE: Backend<ZnxWord = i64>> SvpDerivedDefault<BE> for BE {}
