#![allow(clippy::too_many_arguments)]

//! Independent scalar backends for correctness testing, not production execution.
//!
//! [`FFT64Oracle`] and [`NTT4x30Oracle`] implement the required HAL operations
//! directly, with plain loops, and inherit every optional one. They own their
//! transforms and tables and do not depend on any production CPU backend.
//! `enable-core` registers the generic Core compositions.

mod backend;
#[cfg(feature = "enable-core")]
mod core_impl;
mod embed;
mod family;
mod fft;
mod hal;
mod limbs;
mod normalize;
mod ntt;
#[cfg(feature = "enable-core")]
mod sampling;
#[cfg(feature = "enable-core")]
mod scalar_znx_fill;

#[cfg(test)]
mod tests;

pub use backend::{FFT64Oracle, Handle, NTT4x30Oracle, Oracle};
pub use family::DFTFamily;
pub use fft::Fft64;
pub use ntt::{Ntt4x30, Primes30};

#[cfg(feature = "enable-core")]
pub(crate) use scalar_znx_fill::ScalarZnxFill;
