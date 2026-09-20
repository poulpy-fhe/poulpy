//! Kernel parity tests shared by CPU backend crates.

pub mod normalization;
pub mod normalization_i128;
pub mod ntt;
pub mod reim_conversion;

/// Optional portable FFT comparison adapter for core encryption parity.
#[cfg(feature = "enable-core")]
pub use crate::hal_impl::delegating_backend::ControlledSamplingFFT64Ref;

#[cfg(feature = "enable-core")]
pub mod controlled_sampling;
