//! Optional controlled draws for encryption parity between arbitrary backends.
//!
//! A caller-selected comparison backend may opt into the tested backend's
//! realized samples. This preserves each backend's freedom to map a seed to its
//! own random stream without selecting a particular comparison implementation.
use crate::{Distribution, NoiseInfos, oep::SamplingImpl};
use poulpy_hal::{
    api::*,
    layouts::*,
    oep::{HalVecZnxBigImpl, HalVecZnxImpl},
    test_suite::{download_scalar_znx, download_vec_znx, scalar_znx_backend_mut, vec_znx_backend_mut},
};
use std::sync::{Arc, Mutex, RwLock};

trait SampleProvider: Send + Sync {
    fn scalar(&self, n: usize, dist: Distribution, seed: [u8; 32]) -> Vec<i64>;
    fn noise(&self, n: usize, base2k: usize, noise: NoiseInfos, seed: [u8; 32], big: bool) -> Vec<i64>;
}
struct BackendSamples<B: Backend>(Module<B>);
impl<B> SampleProvider for BackendSamples<B>
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
// The provider is process-wide, not thread-local: a backend is free to sample
// from a worker thread, and a scope that only the calling thread could see would
// turn that into a panic. `ACTIVE` keeps concurrently running test functions from
// overwriting each other's scope.
static ACTIVE: Mutex<()> = Mutex::new(());
static SAMPLES: RwLock<Option<Arc<dyn SampleProvider>>> = RwLock::new(None);

struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        *SAMPLES.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }
}

/// Exposes backend `B`'s realized samples to a comparison backend's sampling adapter.
///
/// The caller supplies the module the draws are taken on and gets it back for the
/// test body. The scope is cleared on return or unwind, and other scopes wait for
/// it, so scopes do not nest. It changes no production sampler; an adapter opts in
/// by calling [`scalar_samples`] and [`noise_samples`]. The backend under test
/// keeps its own sampling and encryption.
pub fn with_backend_samples<B, R>(tested: Module<B>, test: impl FnOnce(&Module<B>) -> R) -> R
where
    B: Backend<ZnxWord = i64> + HalVecZnxImpl + HalVecZnxBigImpl + SamplingImpl + 'static,
{
    let _active = ACTIVE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let samples: Arc<BackendSamples<B>> = Arc::new(BackendSamples::<B>(tested));
    *SAMPLES.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(samples.clone() as Arc<dyn SampleProvider>);
    let _scope = Scope;
    test(&samples.0)
}

fn provider() -> Arc<dyn SampleProvider> {
    SAMPLES
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .expect("controlled sampling needs with_backend_samples")
        .clone()
}
/// Returns a scalar draw from the backend selected by [`with_backend_samples`].
/// A comparison backend's sampling adapter writes these values into its own storage.
///
/// # Panics
/// Panics outside a [`with_backend_samples`] scope.
pub fn scalar_samples(n: usize, dist: Distribution, seed: [u8; 32]) -> Vec<i64> {
    provider().scalar(n, dist, seed)
}

/// Returns the integer noise draws from the selected backend's ordinary (`big =
/// false`) or wide (`big = true`) sampler, before radix placement.
/// The comparison adapter adds these draws at [`NoiseInfos::target_limb_and_shift`].
///
/// # Panics
/// Panics outside a [`with_backend_samples`] scope.
pub fn noise_samples(n: usize, base2k: usize, noise: NoiseInfos, seed: [u8; 32], big: bool) -> Vec<i64> {
    provider().noise(n, base2k, noise, seed, big)
}

/// Registers encryption parity for a caller-selected comparison and tested backend.
///
/// The pair must receive identical realized random draws. Backends with matching
/// streams can be selected directly; otherwise select a sampling adapter that
/// uses [`scalar_samples`] and [`noise_samples`]. No comparison backend is fixed
/// by this suite. Sampling distributions are validated separately.
#[macro_export]
macro_rules! core_encryption_parity_test_suite {
    (mod $name:ident, backend_ref = $backend_ref:ty, backend_test = $backend_test:ty) => {
        mod $name {
            #[test]
            fn glwe_encryption() {
                use ::poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};
                use $crate::test_suite::parity::{ParityShapes, test_glwe_encryption_parity};
                $crate::test_suite::parity::controlled_sampling::with_backend_samples(
                    Module::<$backend_test>::new(256),
                    |tested| {
                        let reference = Module::<$backend_ref>::new(256);
                        test_glwe_encryption_parity(
                            &TestParams {
                                size: 256,
                                n: 256,
                                base2k: 12,
                            },
                            &ParityShapes::default(),
                            &reference,
                            tested,
                        );
                    },
                );
            }
            #[test]
            fn key_encryption() {
                use ::poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};
                use $crate::test_suite::parity::{ParityShapes, test_key_encryption_parity};
                $crate::test_suite::parity::controlled_sampling::with_backend_samples(
                    Module::<$backend_test>::new(256),
                    |tested| {
                        let reference = Module::<$backend_ref>::new(256);
                        test_key_encryption_parity(
                            &TestParams {
                                size: 256,
                                n: 256,
                                base2k: 12,
                            },
                            &ParityShapes::default(),
                            &reference,
                            tested,
                        );
                    },
                );
            }
            #[test]
            fn lwe_encryption() {
                use ::poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};
                use $crate::test_suite::parity::{ParityShapes, test_lwe_encryption_parity};
                $crate::test_suite::parity::controlled_sampling::with_backend_samples(
                    Module::<$backend_test>::new(256),
                    |tested| {
                        let reference = Module::<$backend_ref>::new(256);
                        test_lwe_encryption_parity(
                            &TestParams {
                                size: 256,
                                n: 256,
                                base2k: 12,
                            },
                            &ParityShapes::default(),
                            &reference,
                            tested,
                        );
                    },
                );
            }
        }
    };
}
