//! Kernel parity tests shared by CPU backend crates.

pub mod ntt;
pub mod reim_conversion;
pub mod znx_normalize;

pub mod conjugate_invariant;
/// Optional portable FFT comparison adapter, on both rings, for encryption parity.
/// Its sampler answers from the controlled-sampling scope and panics outside
/// one, so it is gated on `enable-test-suite` rather than on `enable-core`.
#[cfg(any(all(test, feature = "enable-core"), feature = "enable-test-suite"))]
pub use crate::hal_impl::delegating_backend::ControlledSamplingFFT64Portable;

#[cfg(any(all(test, feature = "enable-core"), feature = "enable-test-suite"))]
pub mod controlled_sampling;
