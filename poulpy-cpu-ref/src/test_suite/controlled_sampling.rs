//! Optional FFT64 comparison adapter for controlled encryption parity.
//!
//! The generic parity suite and sample provider live in `poulpy-core`; callers
//! may select this adapter, another adapter, or backends with matching streams.
use super::ControlledSamplingFFT64Ref;
use poulpy_core::{
    Distribution, NoiseInfos,
    oep::SamplingImpl,
    test_suite::parity::controlled_sampling::{noise_samples, scalar_samples},
};
use poulpy_hal::layouts::*;

// Safety: this test reference backend copies distribution-correct draws from
// the backend under test and mutates only the selected coefficient/limb column.
unsafe impl SamplingImpl for ControlledSamplingFFT64Ref {
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
    fn vec_znx_add_normal(
        _: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    ) {
        let samples = noise_samples(res.n(), base2k, noise, seed, false);
        let (limb, shift) = noise.target_limb_and_shift(base2k);
        for (dst, sample) in res.at_mut(col, limb).iter_mut().zip(samples) {
            *dst += sample << shift;
        }
    }
    fn vec_znx_big_add_normal(
        _: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    ) {
        let samples = noise_samples(res.n(), base2k, noise, seed, true);
        let (limb, shift) = noise.target_limb_and_shift(base2k);
        for (dst, sample) in res.at_mut(col, limb).iter_mut().zip(samples) {
            *dst += sample << shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Ref, hal_impl::delegating_backend::DifferentSamplingFFT64Ref};
    use poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples;

    fn changed_seed(mut seed: [u8; 32]) -> [u8; 32] {
        seed[0] ^= 0x5A;
        seed
    }
    // Safety: forwarding to the same correct samplers with a bijection on seeds
    // preserves their distribution, buffer bounds, and per-backend determinism.
    unsafe impl SamplingImpl for DifferentSamplingFFT64Ref {
        fn scalar_znx_fill_distribution(
            module: &Module<Self>,
            res: &mut ScalarZnxBackendMut<'_, Self>,
            col: usize,
            dist: Distribution,
            seed: [u8; 32],
        ) {
            FFT64Ref::scalar_znx_fill_distribution(module.reinterpret(), res, col, dist, changed_seed(seed));
        }
        fn vec_znx_add_normal(
            module: &Module<Self>,
            base2k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            col: usize,
            noise: NoiseInfos,
            seed: [u8; 32],
        ) {
            FFT64Ref::vec_znx_add_normal(module.reinterpret(), base2k, res, col, noise, changed_seed(seed));
        }
        fn vec_znx_big_add_normal(
            module: &Module<Self>,
            base2k: usize,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            col: usize,
            noise: NoiseInfos,
            seed: [u8; 32],
        ) {
            FFT64Ref::vec_znx_big_add_normal(
                module.reinterpret(),
                base2k,
                &mut res.reborrow_backend_mut().into_backend::<FFT64Ref>(),
                col,
                noise,
                changed_seed(seed),
            );
        }
    }
    #[test]
    fn encryption_parity_accepts_different_backend_random_streams() {
        let original = with_backend_samples(Module::<FFT64Ref>::new(64), |_| {
            scalar_samples(64, Distribution::TernaryProb(0.5), [7; 32])
        });
        let different = with_backend_samples(Module::<DifferentSamplingFFT64Ref>::new(64), |_| {
            scalar_samples(64, Distribution::TernaryProb(0.5), [7; 32])
        });
        assert_ne!(original, different);
        with_backend_samples(Module::<DifferentSamplingFFT64Ref>::new(64), |tested| {
            let reference = Module::<ControlledSamplingFFT64Ref>::new(64);
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
