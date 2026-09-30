use super::FFT64Portable;
use super::NTT4x30Portable;
use poulpy_hal::layouts::Ring;

use crate::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ConvolutionDefault, NTT4x30ModuleDefault, NTT4x30SvpDefault, NTT4x30VecZnxBigDefault,
    NTT4x30VecZnxDftDefault, NTT4x30VmpDefault,
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64Portable<poulpy_hal::layouts::ConjugateInvariant> {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for FFT64Portable<R>
where
    Self: crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx!();
}

unsafe impl<R: Ring> HalModuleImpl for FFT64Portable<R>
where
    crate::reference::fft64::module::FFT64Plan<f64, R>: crate::reference::fft64::module::FFT64PlanNew,
{
    crate::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl<R: Ring> HalVmpImpl for FFT64Portable<R>
where
    Self: crate::reference::fft64::ring_arith::Fft64RingArith + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl<R: Ring> HalConvolutionImpl for FFT64Portable<R>
where
    Self: crate::reference::fft64::ring_arith::Fft64RingArith + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl<R: Ring> HalVecZnxBigImpl for FFT64Portable<R>
where
    Self: crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl<R: Ring> HalSvpImpl for FFT64Portable<R>
where
    Self: crate::reference::fft64::ring_arith::Fft64RingArith + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl<R: Ring> HalVecZnxDftImpl for FFT64Portable<R>
where
    Self: crate::reference::fft64::ring_arith::Fft64RingArith + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30Portable<poulpy_hal::layouts::ConjugateInvariant> {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for NTT4x30Portable<R>
where
    Self: crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx!();
}

unsafe impl<R: Ring> HalModuleImpl for NTT4x30Portable<R>
where
    crate::reference::ntt4x30::vec_znx_dft::NttPlan<crate::reference::ntt4x30::primes::Primes30, R>:
        crate::reference::ntt4x30::vec_znx_dft::NttPlanNew,
{
    crate::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl<R: Ring> HalVmpImpl for NTT4x30Portable<R>
where
    Self: crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTable<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTableInv<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vmp!(NTT4x30VmpDefault);
}

unsafe impl<R: Ring> HalConvolutionImpl for NTT4x30Portable<R>
where
    Self: crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTable<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTableInv<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_convolution!(NTT4x30ConvolutionDefault);

    fn cnv_apply_dft_sum_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault>::cnv_apply_dft_sum_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'_, Self>],
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault>::cnv_apply_dft_sum_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            terms,
            &mut scratch,
        );
    }
}

unsafe impl<R: Ring> HalVecZnxBigImpl for NTT4x30Portable<R>
where
    Self: crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl<R: Ring> HalSvpImpl for NTT4x30Portable<R>
where
    Self: crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTable<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTableInv<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_svp!(NTT4x30SvpDefault);
}

unsafe impl<R: Ring> HalVecZnxDftImpl for NTT4x30Portable<R>
where
    Self: crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTable<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::ntt4x30::NttDFTExecute<crate::reference::ntt4x30::ntt::NttTableInv<crate::reference::ntt4x30::primes::Primes30, R>>
        + crate::reference::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
