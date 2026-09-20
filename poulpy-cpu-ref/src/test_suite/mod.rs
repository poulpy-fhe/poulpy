//! Kernel parity tests shared by CPU backend crates.

pub mod normalization;
pub mod normalization_i128;
pub mod ntt;
pub mod reim_conversion;

/// Portable FFT backend whose sampling can be controlled by the core parity harness.
#[cfg(feature = "enable-core")]
pub use crate::hal_impl::delegating_backend::ControlledSamplingFFT64Ref;

#[cfg(feature = "enable-core")]
pub mod controlled_sampling;
