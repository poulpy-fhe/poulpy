use crate::{
    FFT64Oracle, NTT4x30Oracle,
    hal_defaults::{
        FFT64ModuleDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, HalVecZnxDefault, NTT4x30ModuleDefault,
        NTT4x30VecZnxBigDefault, NTT4x30VecZnxDftDefault,
    },
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalModuleImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
};

#[macro_use]
mod vec_znx;
#[macro_use]
mod module;
#[macro_use]
mod vec_znx_big;
#[macro_use]
mod vec_znx_dft;

unsafe impl HalVecZnxImpl for FFT64Oracle {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl for FFT64Oracle {
    hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVecZnxBigImpl for FFT64Oracle {
    hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalVecZnxDftImpl for FFT64Oracle {
    hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl for NTT4x30Oracle {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl for NTT4x30Oracle {
    hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVecZnxBigImpl for NTT4x30Oracle {
    hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalVecZnxDftImpl for NTT4x30Oracle {
    hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
