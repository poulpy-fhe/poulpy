//! HAL registrations for [`FFT64CIRef`](super::FFT64CIRef).

use super::FFT64CIRef;

use crate::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault,
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl HalVecZnxImpl for FFT64CIRef {
    crate::hal_impl_vec_znx!(fft64);
}

unsafe impl HalModuleImpl for FFT64CIRef {
    crate::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl for FFT64CIRef {
    crate::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl for FFT64CIRef {
    crate::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl for FFT64CIRef {
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl for FFT64CIRef {
    crate::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl for FFT64CIRef {
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}
