//! Kernel parity tests shared by CPU backend crates.

pub mod normalization;
pub mod normalization_i128;
pub mod ntt;
pub mod reim_conversion;

pub mod conjugate_invariant;
/// Optional portable FFT comparison adapter for core encryption parity.
/// Its sampler answers from the controlled-sampling scope and panics outside
/// one, so it is gated on `enable-test-suite` rather than on `enable-core`.
#[cfg(any(all(test, feature = "enable-core"), feature = "enable-test-suite"))]
pub use crate::hal_impl::delegating_backend::ControlledSamplingFFT64Ref;

#[cfg(any(all(test, feature = "enable-core"), feature = "enable-test-suite"))]
pub mod controlled_sampling;
