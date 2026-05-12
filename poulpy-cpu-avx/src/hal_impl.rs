use crate::{FFT64Avx, NTT120Avx};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT120ConvolutionDefault, NTT120ModuleDefault, NTT120SvpDefault, NTT120VecZnxBigDefault,
    NTT120VecZnxDftDefault, NTT120VmpDefault,
};
use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, Module, NoiseInfos, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_module!(NTT120ModuleDefault);
}

unsafe impl HalVmpImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vmp!(NTT120VmpDefault);
}

unsafe impl HalConvolutionImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_convolution!(NTT120ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT120VecZnxBigDefault);
}

unsafe impl HalSvpImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_svp!(NTT120SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT120VecZnxDftDefault);
}
