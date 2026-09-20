//! Sampling oracle for encryption parity. Only test code installs this scope.
//!
//! The reference composition receives the tested backend's realized samples.
//! This preserves each backend's freedom to map a seed to its own random stream.
use super::ControlledSamplingFFT64Ref;
use poulpy_core::{Distribution, NoiseInfos, oep::SamplingImpl};
use poulpy_hal::{
    api::*,
    layouts::*,
    oep::{HalModuleImpl, HalVecZnxBigImpl, HalVecZnxImpl},
    test_suite::{download_scalar_znx, download_vec_znx, scalar_znx_backend_mut, vec_znx_backend_mut},
};
use std::{cell::RefCell, rc::Rc};

trait SampleOracle {
    fn scalar(&self, n: usize, dist: Distribution, seed: [u8; 32]) -> Vec<i64>;
    fn noise(&self, n: usize, base2k: usize, noise: NoiseInfos, seed: [u8; 32], big: bool) -> Vec<i64>;
}
struct BackendSamples<B: Backend>(Module<B>);
impl<B> SampleOracle for BackendSamples<B>
where
    B: Backend<ZnxWord = i64> + HalVecZnxImpl + HalVecZnxBigImpl + SamplingImpl,
{
    fn scalar(&self, n: usize, dist: Distribution, seed: [u8; 32]) -> Vec<i64> {
        let mut out = self.0.scalar_znx_alloc(n, 1);
        B::scalar_znx_fill_distribution(&self.0, &mut scalar_znx_backend_mut::<B>(&mut out), 0, dist, seed);
        download_scalar_znx::<B>(&out).at(0, 0).to_vec()
    }
    fn noise(&self, n: usize, base2k: usize, noise: NoiseInfos, seed: [u8; 32], big: bool) -> Vec<i64> {
        let size = noise.k.div_ceil(base2k);
        let mut out = self.0.vec_znx_alloc(n, 1, size);
        self.0.vec_znx_zero(&mut vec_znx_backend_mut::<B>(&mut out), 0);
        if big {
            let mut wide = self.0.vec_znx_big_alloc(n, 1, size);
            // From-small initializes every limb in the backend's own representation.
            self.0.vec_znx_big_from_small(
                &mut wide.to_backend_mut(),
                0,
                &VecZnxToBackendRef::<B>::to_backend_ref(&out),
                0,
            );
            B::vec_znx_big_add_normal(&self.0, base2k, &mut wide.to_backend_mut(), 0, noise, seed);
            let mut scratch = ScratchOwned::<B>::alloc(self.0.vec_znx_big_normalize_tmp_bytes());
            self.0.vec_znx_big_normalize(
                &mut VecZnxToBackendMut::<B>::to_backend_mut(&mut out),
                base2k,
                size * base2k,
                0,
                0,
                &wide.to_backend_ref(),
                base2k,
                0,
                &mut scratch.borrow(),
            );
        } else {
            B::vec_znx_add_normal(&self.0, base2k, &mut vec_znx_backend_mut::<B>(&mut out), 0, noise, seed);
        }
        let host = download_vec_znx::<B>(&out);
        // Recover the small integer draw exactly from its radix representation.
        (0..n)
            .map(|i| {
                let mut sample = 0i128;
                for limb in 0..size {
                    let value = host.at(0, limb)[i] as i128;
                    let exponent = noise.k as i64 - ((limb + 1) * base2k) as i64;
                    if exponent >= 127 {
                        assert_eq!(value, 0);
                    } else if exponent >= 0 {
                        sample += value << exponent;
                    } else {
                        sample += value >> -exponent;
                    }
                }
                i64::try_from(sample).expect("parity noise sample fits an integer")
            })
            .collect()
    }
}
thread_local! {
    static ORACLE: RefCell<Option<Rc<dyn SampleOracle>>> = RefCell::new(None);
}
struct Restore(Option<Rc<dyn SampleOracle>>);
impl Drop for Restore {
    fn drop(&mut self) {
        ORACLE.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}
/// Executes a parity test with a thread-local, unwind-safe sampling oracle.
/// The backend under test keeps its actual sampling and encryption implementations.
pub fn with_backend_samples<B, R>(n: usize, test: impl FnOnce() -> R) -> R
where
    B: Backend<ZnxWord = i64> + HalModuleImpl + HalVecZnxImpl + HalVecZnxBigImpl + SamplingImpl + 'static,
{
    let oracle: Rc<dyn SampleOracle> = Rc::new(BackendSamples::<B>(Module::<B>::new(n as u64)));
    let _restore = Restore(ORACLE.with(|slot| slot.replace(Some(oracle))));
    test()
}
fn oracle() -> Rc<dyn SampleOracle> {
    ORACLE.with(|slot| {
        slot.borrow()
            .as_ref()
            .expect("controlled sampling needs with_backend_samples")
            .clone()
    })
}
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
        let samples = oracle().scalar(res.n(), dist, seed);
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
        let samples = oracle().noise(res.n(), base2k, noise, seed, false);
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
        let samples = oracle().noise(res.n(), base2k, noise, seed, true);
        let (limb, shift) = noise.target_limb_and_shift(base2k);
        for (dst, sample) in res.at_mut(col, limb).iter_mut().zip(samples) {
            *dst += sample << shift;
        }
    }
}

/// Registers randomized-operation parity against the sample-injected CPU reference.
#[macro_export]
macro_rules! core_encryption_parity_test_suite {
    (mod $name:ident, backend = $backend:ty) => {
        mod $name {
            #[test]
            fn glwe_encryption() {
                use ::poulpy_core::test_suite::parity::{ParityShapes, test_glwe_encryption_parity};
                use ::poulpy_hal::{layouts::Module, test_suite::TestParams};
                $crate::test_suite::controlled_sampling::with_backend_samples::<$backend, _>(256, || {
                    let reference = Module::<$crate::test_suite::ControlledSamplingFFT64Ref>::new(256);
                    let tested = Module::<$backend>::new(256);
                    test_glwe_encryption_parity(
                        &TestParams {
                            size: 256,
                            n: 256,
                            base2k: 12,
                        },
                        &ParityShapes::default(),
                        &reference,
                        &tested,
                    );
                });
            }
            #[test]
            fn key_encryption() {
                use ::poulpy_core::test_suite::parity::{ParityShapes, test_key_encryption_parity};
                use ::poulpy_hal::{layouts::Module, test_suite::TestParams};
                $crate::test_suite::controlled_sampling::with_backend_samples::<$backend, _>(256, || {
                    let reference = Module::<$crate::test_suite::ControlledSamplingFFT64Ref>::new(256);
                    let tested = Module::<$backend>::new(256);
                    test_key_encryption_parity(
                        &TestParams {
                            size: 256,
                            n: 256,
                            base2k: 12,
                        },
                        &ParityShapes::default(),
                        &reference,
                        &tested,
                    );
                });
            }
            #[test]
            fn lwe_encryption() {
                use ::poulpy_core::test_suite::parity::{ParityShapes, test_lwe_encryption_parity};
                use ::poulpy_hal::{layouts::Module, test_suite::TestParams};
                $crate::test_suite::controlled_sampling::with_backend_samples::<$backend, _>(256, || {
                    let reference = Module::<$crate::test_suite::ControlledSamplingFFT64Ref>::new(256);
                    let tested = Module::<$backend>::new(256);
                    test_lwe_encryption_parity(
                        &TestParams {
                            size: 256,
                            n: 256,
                            base2k: 12,
                        },
                        &ParityShapes::default(),
                        &reference,
                        &tested,
                    );
                });
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Ref, hal_impl::delegating_backend::DifferentSamplingFFT64Ref};

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
        let original = BackendSamples(Module::<FFT64Ref>::new(64));
        let different = BackendSamples(Module::<DifferentSamplingFFT64Ref>::new(64));
        assert_ne!(
            original.scalar(64, Distribution::TernaryProb(0.5), [7; 32]),
            different.scalar(64, Distribution::TernaryProb(0.5), [7; 32])
        );
        with_backend_samples::<DifferentSamplingFFT64Ref, _>(64, || {
            let reference = Module::<ControlledSamplingFFT64Ref>::new(64);
            let tested = Module::<DifferentSamplingFFT64Ref>::new(64);
            let params = poulpy_hal::test_suite::TestParams {
                size: 64,
                n: 64,
                base2k: 12,
            };
            let shapes = poulpy_core::test_suite::parity::ParityShapes {
                ranks: vec![1],
                dsizes: None,
            };
            poulpy_core::test_suite::parity::test_glwe_encryption_parity(&params, &shapes, &reference, &tested);
            poulpy_core::test_suite::parity::test_lwe_encryption_parity(&params, &shapes, &reference, &tested);
        });
    }
}
