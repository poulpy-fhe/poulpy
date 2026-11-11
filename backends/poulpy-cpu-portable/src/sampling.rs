//! Host implementation of the `poulpy-core` sampling extension point.

/// Implements [`SamplingImpl`](poulpy_core::oep::SamplingImpl) for a CPU
/// backend with this crate's host [`ScalarZnxFill`](crate::ScalarZnxFill)
/// methods and portable noise kernels, all applied straight to the backend
/// buffer.
///
/// The last argument selects the `VecZnxBig` word family: `fft64` for
/// `BigWord = i64`, `ntt4x30` for `BigWord = i128`. Without a ring argument the
/// macro implements both rings; `impl_sampling_host!(Backend, Ring, family)`
/// implements `Ring` only.
///
/// ```ignore
/// poulpy_cpu_portable::impl_sampling_host!(FFT64Avx, fft64);
/// poulpy_cpu_portable::impl_sampling_host!(NTT4x30Avx, ntt4x30);
/// poulpy_cpu_portable::impl_sampling_host!(NTT3x42IfmaRayon, ::poulpy_hal::layouts::Standard, ntt4x30);
/// ```
#[macro_export]
macro_rules! impl_sampling_host {
    ($be:ty, $family:ident) => {
        $crate::impl_sampling_host!($be, ::poulpy_hal::layouts::Standard, $family);
        $crate::impl_sampling_host!($be, ::poulpy_hal::layouts::ConjugateInvariant, $family);
    };
    ($be:ty, $ring:ty, fft64) => {
        $crate::impl_sampling_host!(
            @impl $be,
            $ring,
            $crate::kernels::fft64::vec_znx_big::vec_znx_big_add_noise_portable::<_, $be, $ring>
        );
    };
    ($be:ty, $ring:ty, ntt4x30) => {
        $crate::impl_sampling_host!(
            @impl $be,
            $ring,
            $crate::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_noise_portable::<_, $be, $ring>
        );
    };
    (@impl $be:ty, $ring:ty, $big_kernel:expr) => {
        unsafe impl ::poulpy_core::oep::SamplingImpl<$ring> for $be {
            fn scalar_znx_fill_distribution(
                _module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::ScalarZnxBackendMut<'_, $be, $ring>,
                res_col: usize,
                dist: ::poulpy_core::Distribution,
                seed: [u8; 32],
            ) {
                use ::poulpy_hal::layouts::{ZnxViewMut, ZnxWord};
                use $crate::ScalarZnxFill;
                let mut source = ::poulpy_hal::source::Source::new(seed);
                match dist {
                    ::poulpy_core::Distribution::TernaryFixed(hw) => res.fill_ternary_hw(res_col, hw, &mut source),
                    ::poulpy_core::Distribution::TernaryProb(prob) => res.fill_ternary_prob(res_col, prob, &mut source),
                    ::poulpy_core::Distribution::BinaryFixed(hw) => res.fill_binary_hw(res_col, hw, &mut source),
                    ::poulpy_core::Distribution::BinaryProb(prob) => res.fill_binary_prob(res_col, prob, &mut source),
                    ::poulpy_core::Distribution::BinaryBlock(block_size) => {
                        res.fill_binary_block(res_col, block_size, &mut source)
                    }
                    ::poulpy_core::Distribution::ZERO => {
                        res.at_mut(res_col, 0)
                            .fill(<<$be as ::poulpy_hal::layouts::Backend>::ZnxWord as ZnxWord>::from_i64(
                                0,
                            ))
                    }
                    ::poulpy_core::Distribution::NONE | ::poulpy_core::Distribution::ENCAPSULATED(_) => {
                        panic!("scalar_znx_fill_distribution: {dist:?} is not a sampleable distribution")
                    }
                }
            }

            fn vec_znx_add_noise(
                module: &::poulpy_hal::layouts::Module<$be>,
                base2k: usize,
                k: usize,
                res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, $be, $ring>,
                res_col: usize,
                noise: ::poulpy_core::Noise,
                seed: [u8; 32],
            ) {
                assert!(res.n().is_power_of_two() && res.n() <= module.n(), "noise degree outside module");
                let mut source = ::poulpy_hal::source::Source::new(seed);
                $crate::kernels::vec_znx::vec_znx_add_noise_portable::<$be, $ring>(
                    base2k,
                    k,
                    res,
                    res_col,
                    noise,
                    &mut source,
                );
            }

            fn vec_znx_big_add_noise(
                module: &::poulpy_hal::layouts::Module<$be>,
                base2k: usize,
                k: usize,
                mut res: &mut ::poulpy_hal::layouts::VecZnxBigBackendMut<'_, $be, $ring>,
                res_col: usize,
                noise: ::poulpy_core::Noise,
                seed: [u8; 32],
            ) {
                assert!(res.n().is_power_of_two() && res.n() <= module.n(), "noise degree outside module");
                let mut source = ::poulpy_hal::source::Source::new(seed);
                $big_kernel(
                    base2k,
                    k,
                    &mut res,
                    res_col,
                    noise,
                    &mut source,
                );
            }
        }
    };
}
