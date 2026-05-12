use crate::{
    FFT64Ref, NTT120Ref,
    hal_defaults::{
        FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault,
        FFT64VmpDefault, HalVecZnxDefault, NTT120ConvolutionDefault, NTT120ModuleDefault, NTT120SvpDefault,
        NTT120VecZnxBigDefault, NTT120VecZnxDftDefault, NTT120VmpDefault,
    },
};
use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, Module, NoiseInfos, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

#[macro_use]
mod vec_znx;
#[macro_use]
mod module;
#[macro_use]
mod vmp;
#[macro_use]
mod convolution;
#[macro_use]
mod vec_znx_big;
#[macro_use]
mod svp;
#[macro_use]
mod vec_znx_dft;
#[cfg(all(test, feature = "enable-core"))]
pub(crate) mod delegating_backend;

unsafe impl HalVecZnxImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Ref> for FFT64Ref {
    hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<FFT64Ref> for FFT64Ref {
    hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Ref> for NTT120Ref {
    hal_impl_module!(NTT120ModuleDefault);
}

unsafe impl HalVmpImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vmp!(NTT120VmpDefault);
}

unsafe impl HalConvolutionImpl<NTT120Ref> for NTT120Ref {
    hal_impl_convolution!(NTT120ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx_big!(NTT120VecZnxBigDefault);
}

unsafe impl HalSvpImpl<NTT120Ref> for NTT120Ref {
    hal_impl_svp!(NTT120SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx_dft!(NTT120VecZnxDftDefault);
}
