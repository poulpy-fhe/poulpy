//! Host implementation of the `poulpy-core` sampling extension point.

macro_rules! impl_sampling_host {
    ($be:ty, fft64) => {
        $crate::sampling::impl_sampling_host!(@impl $be, $crate::reference::fft64::vec_znx_big::vec_znx_big_add_normal_ref::<_, $be>);
    };
    ($be:ty, ntt4x30) => {
        $crate::sampling::impl_sampling_host!(@impl $be, $crate::reference::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_normal_ref::<_, $be>);
    };
    (@impl $be:ty, $big_kernel:expr) => {
        unsafe impl ::poulpy_core::oep::SamplingImpl for $be {
            fn scalar_znx_fill_distribution(
                _module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::ScalarZnxBackendMut<'_, $be>,
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

            fn vec_znx_add_normal(
                _module: &::poulpy_hal::layouts::Module<$be>,
                base2k: usize,
                res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, $be>,
                res_col: usize,
                noise: ::poulpy_core::NoiseInfos,
                seed: [u8; 32],
            ) {
                let mut source = ::poulpy_hal::source::Source::new(seed);
                $crate::reference::vec_znx::vec_znx_add_normal_ref::<$be>(
                    base2k,
                    res,
                    res_col,
                    noise.k,
                    noise.sigma,
                    noise.bound,
                    &mut source,
                );
            }

            fn vec_znx_big_add_normal(
                _module: &::poulpy_hal::layouts::Module<$be>,
                base2k: usize,
                mut res: &mut ::poulpy_hal::layouts::VecZnxBigBackendMut<'_, $be>,
                res_col: usize,
                noise: ::poulpy_core::NoiseInfos,
                seed: [u8; 32],
            ) {
                let mut source = ::poulpy_hal::source::Source::new(seed);
                $big_kernel(
                    base2k,
                    &mut res,
                    res_col,
                    noise.k,
                    noise.sigma,
                    noise.bound,
                    &mut source,
                );
            }
        }
    };
}

pub(crate) use impl_sampling_host;
