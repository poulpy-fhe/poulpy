//! Optional FFT64 comparison adapters, one per ring, for controlled encryption parity.
//!
//! The generic parity suite and sample provider live in `poulpy-core`; callers
//! may select this adapter, another adapter, or backends with matching streams.
use super::{ControlledSamplingFFT64CIPortable, ControlledSamplingFFT64Portable};
use poulpy_core::{
    Distribution, Noise,
    oep::SamplingImpl,
    test_suite::parity::controlled_sampling::{noise_samples, scalar_samples},
};
use poulpy_hal::layouts::*;

// Safety: these test comparison backends copy distribution-correct draws from
// the backend under test and mutate only the selected coefficient/limb column.
macro_rules! impl_controlled_sampling {
    ($($be:ty),+) => {$(
        unsafe impl SamplingImpl for $be {
            fn scalar_znx_fill_distribution(
                _: &Module<Self>,
                res: &mut ScalarZnxBackendMut<'_, Self>,
                col: usize,
                dist: Distribution,
                seed: [u8; 32],
            ) {
                let samples = scalar_samples(res.n(), dist, seed);
                res.at_mut(col, 0).copy_from_slice(&samples);
            }
            fn vec_znx_add_noise(
                _: &Module<Self>,
                base2k: usize,
                k: usize,
                res: &mut VecZnxBackendMut<'_, Self>,
                col: usize,
                noise: Noise,
                seed: [u8; 32],
            ) {
                noise.validate();
                assert!((1..=63).contains(&base2k));
                assert!(k > 0 && k.div_ceil(base2k) <= res.size());
                assert!(col < res.cols());
                let samples = noise_samples(res.n(), base2k, k, noise, seed, false);
                for (limb, digits) in samples.chunks(res.n()).enumerate() {
                    for (dst, digit) in res.at_mut(col, limb).iter_mut().zip(digits) {
                        *dst = dst.wrapping_add(*digit);
                    }
                }
            }
            fn vec_znx_big_add_noise(
                _: &Module<Self>,
                base2k: usize,
                k: usize,
                res: &mut VecZnxBigBackendMut<'_, Self>,
                col: usize,
                noise: Noise,
                seed: [u8; 32],
            ) {
                noise.validate();
                assert!((1..=63).contains(&base2k));
                assert!(k > 0 && k.div_ceil(base2k) <= res.size());
                assert!(col < res.cols());
                let samples = noise_samples(res.n(), base2k, k, noise, seed, true);
                for (limb, digits) in samples.chunks(res.n()).enumerate() {
                    for (dst, digit) in res.at_mut(col, limb).iter_mut().zip(digits) {
                        *dst = dst.wrapping_add(*digit);
                    }
                }
            }
        }
    )+};
}

impl_controlled_sampling!(ControlledSamplingFFT64Portable, ControlledSamplingFFT64CIPortable);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Portable, hal_impl::delegating_backend::DifferentSamplingFFT64Portable};
    use poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples;

    fn changed_seed(mut seed: [u8; 32]) -> [u8; 32] {
        seed[0] ^= 0x5A;
        seed
    }
    // Safety: forwarding to the same correct samplers with a bijection on seeds
    // preserves their distribution, buffer bounds, and per-backend determinism.
    unsafe impl SamplingImpl for DifferentSamplingFFT64Portable {
        fn scalar_znx_fill_distribution(
            module: &Module<Self>,
            res: &mut ScalarZnxBackendMut<'_, Self>,
            col: usize,
            dist: Distribution,
            seed: [u8; 32],
        ) {
            FFT64Portable::scalar_znx_fill_distribution(module.reinterpret(), res, col, dist, changed_seed(seed));
        }
        fn vec_znx_add_noise(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            col: usize,
            noise: Noise,
            seed: [u8; 32],
        ) {
            FFT64Portable::vec_znx_add_noise(module.reinterpret(), base2k, k, res, col, noise, changed_seed(seed));
        }
        fn vec_znx_big_add_noise(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            col: usize,
            noise: Noise,
            seed: [u8; 32],
        ) {
            FFT64Portable::vec_znx_big_add_noise(
                module.reinterpret(),
                base2k,
                k,
                &mut res.reborrow_backend_mut().into_backend::<FFT64Portable>(),
                col,
                noise,
                changed_seed(seed),
            );
        }
    }
    #[test]
    fn encryption_parity_accepts_different_backend_random_streams() {
        let original = with_backend_samples(Module::<FFT64Portable>::new(64), |_| {
            scalar_samples(64, Distribution::TernaryProb(0.5), [7; 32])
        });
        let different = with_backend_samples(Module::<DifferentSamplingFFT64Portable>::new(64), |_| {
            scalar_samples(64, Distribution::TernaryProb(0.5), [7; 32])
        });
        assert_ne!(original, different);
        with_backend_samples(Module::<DifferentSamplingFFT64Portable>::new(64), |tested| {
            let reference = Module::<ControlledSamplingFFT64Portable>::new(64);
            let params = poulpy_hal::test_suite::TestParams {
                size: 64,
                n: 64,
                base2k: 12,
            };
            let shapes = poulpy_core::test_suite::parity::ParityShapes {
                ranks: vec![1],
                dsizes: None,
            };
            poulpy_core::test_suite::parity::test_glwe_encryption_parity(&params, &shapes, &reference, tested);
            poulpy_core::test_suite::parity::test_lwe_encryption_parity(&params, &shapes, &reference, tested);
        });
    }
}
