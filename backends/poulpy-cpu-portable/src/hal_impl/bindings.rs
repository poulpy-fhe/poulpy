use super::FFT64Portable;
use super::NTT4x30Portable;
use poulpy_hal::layouts::Ring;

use crate::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault,
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalCnvImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64Portable {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for FFT64Portable {}

unsafe impl<Rg: Ring> poulpy_hal::oep::HalScalarZnxImpl<Rg> for FFT64Portable where Self: HalVecZnxImpl<Rg> {}
unsafe impl<Rg: Ring> poulpy_hal::oep::HalMatZnxImpl<Rg> for FFT64Portable where Self: HalVecZnxImpl<Rg> {}

unsafe impl<Rg: Ring> HalVecZnxImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vec_znx!(Rg);
}

unsafe impl HalModuleImpl for FFT64Portable {
    crate::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl<Rg: Ring> HalVmpImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith<Rg> + crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vmp!(FFT64VmpDefault, Rg);
}

unsafe impl<Rg: Ring> HalCnvImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith<Rg> + crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_cnv!(FFT64ConvolutionDefault, Rg);
}

unsafe impl<Rg: Ring> HalVecZnxBigImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault, Rg);
}

unsafe impl<Rg: Ring> HalSvpImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith<Rg> + crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_svp!(FFT64SvpDefault, Rg);
}

unsafe impl<Rg: Ring> HalVecZnxDftImpl<Rg> for FFT64Portable
where
    Self: crate::kernels::fft64::ring_arith::Fft64RingArith<Rg> + crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault, Rg);
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30Portable {
    crate::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30Portable {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for NTT4x30Portable {}

unsafe impl<Rg: Ring> poulpy_hal::oep::HalScalarZnxImpl<Rg> for NTT4x30Portable where Self: HalVecZnxImpl<Rg> {}
unsafe impl<Rg: Ring> poulpy_hal::oep::HalMatZnxImpl<Rg> for NTT4x30Portable where Self: HalVecZnxImpl<Rg> {}

unsafe impl<Rg: Ring> HalVecZnxImpl<Rg> for NTT4x30Portable
where
    Self: crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vec_znx!(Rg);
}

unsafe impl HalModuleImpl for NTT4x30Portable {
    crate::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl<Rg: Ring> HalVecZnxBigImpl<Rg> for NTT4x30Portable
where
    Self: crate::kernels::znx::ZnxAutomorphism<Rg>,
{
    crate::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault, Rg);
}
