use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use crate::{
    FFT64Ref,
    hal_defaults::{
        FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault,
        FFT64VmpDefault, HalVecZnxDefault,
    },
};

// The two fixtures reuse the same independently implemented HAL surface. Core
// tests specialize the first marker; randomized parity injects samples into the
// second without modifying a production backend or sharing prepared storage.
macro_rules! impl_fft64_delegating_backend {
    ($be:ident) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $be;

        impl poulpy_hal::execution::ScratchWorkers for $be {}

        poulpy_hal::impl_backend_from!($be, FFT64Ref);

        impl poulpy_hal::layouts::MaxBase2k for $be {
            fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
                <FFT64Ref as poulpy_hal::layouts::MaxBase2k>::max_base2k(n, products, failure_bits, squaring)
            }
        }

        unsafe impl HalVecZnxImpl for $be {
            crate::hal_impl_vec_znx!();
        }

        unsafe impl HalModuleImpl for $be {
            crate::hal_impl_module!(FFT64ModuleDefault);
        }

        unsafe impl HalVmpImpl for $be {
            crate::hal_impl_vmp!(FFT64VmpDefault);
        }

        unsafe impl HalConvolutionImpl for $be {
            crate::hal_impl_convolution!(FFT64ConvolutionDefault);
        }

        unsafe impl HalVecZnxBigImpl for $be {
            crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
        }

        unsafe impl HalSvpImpl for $be {
            crate::hal_impl_svp!(FFT64SvpDefault);
        }

        unsafe impl HalVecZnxDftImpl for $be {
            crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
        }
    };
}

#[cfg(test)]
impl_fft64_delegating_backend!(DelegatingFFT64Ref);
#[cfg(test)]
crate::forward_znx_kernels!(DelegatingFFT64Ref => FFT64Ref);
#[cfg(test)]
crate::forward_fft64_kernels!(DelegatingFFT64Ref => FFT64Ref);
impl_fft64_delegating_backend!(ControlledSamplingFFT64Ref);
crate::forward_znx_kernels!(ControlledSamplingFFT64Ref => FFT64Ref);
crate::forward_fft64_kernels!(ControlledSamplingFFT64Ref => FFT64Ref);

#[test]
fn test_normalization_kernels_delegating_fft64_ref() {
    crate::test_suite::normalization::test_normalization_kernels::<DelegatingFFT64Ref>();
}

poulpy_core::impl_core_reference_full!(ControlledSamplingFFT64Ref);

#[cfg(test)]
impl_fft64_delegating_backend!(DifferentSamplingFFT64Ref);
#[cfg(test)]
crate::forward_znx_kernels!(DifferentSamplingFFT64Ref => FFT64Ref);
#[cfg(test)]
crate::forward_fft64_kernels!(DifferentSamplingFFT64Ref => FFT64Ref);
#[cfg(test)]
poulpy_core::impl_core_reference_full!(DifferentSamplingFFT64Ref);

#[cfg(all(test, feature = "enable-bin-fhe"))]
impl_fft64_delegating_backend!(BinFheOverrideFFT64);
#[cfg(all(test, feature = "enable-bin-fhe"))]
crate::forward_znx_kernels!(BinFheOverrideFFT64 => FFT64Ref);
#[cfg(all(test, feature = "enable-bin-fhe"))]
crate::forward_fft64_kernels!(BinFheOverrideFFT64 => FFT64Ref);
