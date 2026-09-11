//! Host implementation of the `poulpy-core` sampling extension point.

/// Implements [`SamplingImpl`](poulpy_core::oep::SamplingImpl) for a CPU
/// backend with the host `ScalarZnx::fill_*` methods applied straight to the
/// backend buffer — no host round trip.
///
/// ```ignore
/// poulpy_cpu_ref::impl_sampling_host!(FFT64Avx);
/// ```
#[macro_export]
macro_rules! impl_sampling_host {
    ($be:ty) => {
        unsafe impl ::poulpy_core::oep::SamplingImpl<$be> for $be {
            fn scalar_znx_fill_distribution(
                _module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::ScalarZnxBackendMut<'_, $be>,
                res_col: usize,
                dist: ::poulpy_core::Distribution,
                seed: [u8; 32],
            ) {
                use ::poulpy_hal::layouts::{ZnxViewMut, ZnxWord};
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
        }
    };
}
