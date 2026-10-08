use super::FFT64Portable;
use super::NTT4x30Portable;
use poulpy_hal::layouts::Ring;

use crate::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault,
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64Portable {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for FFT64Portable<R>
where
    Self: crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx!();
}

unsafe impl<R: Ring> HalModuleImpl for FFT64Portable<R>
where
    crate::kernels::fft64::module::FFT64Plan<f64, R>: crate::kernels::fft64::module::FFT64PlanNew,
{
    crate::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl<R: Ring> HalVmpImpl for FFT64Portable<R>
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith + crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl<R: Ring> HalConvolutionImpl for FFT64Portable<R>
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith + crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl<R: Ring> HalVecZnxBigImpl for FFT64Portable<R>
where
    Self: crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl<R: Ring> HalSvpImpl for FFT64Portable<R>
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith + crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl<R: Ring> HalVecZnxDftImpl for FFT64Portable<R>
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith + crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30Portable {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for NTT4x30Portable<R>
where
    Self: crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx!();
}

unsafe impl<R: Ring> HalModuleImpl for NTT4x30Portable<R>
where
    crate::kernels::ntt4x30::vec_znx_dft::NttPlan<crate::kernels::ntt4x30::primes::Primes30, R>:
        crate::kernels::ntt4x30::vec_znx_dft::NttPlanNew,
{
    crate::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl<R: Ring> HalVecZnxBigImpl for NTT4x30Portable<R>
where
    Self: crate::kernels::znx::ZnxAutomorphism,
{
    crate::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

