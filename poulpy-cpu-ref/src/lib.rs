#![allow(clippy::too_many_arguments)]

//! Reference (portable) CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two reference implementations for [`poulpy_hal`]:
//!
//! - [`FFT64Ref`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT4x30Ref`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt4x30`] module.
//!
//! Both are canonical reference implementations: portable across all CPU architectures,
//! prioritising correctness and debuggability over throughput.
//!
//! # Features
//!
//! The crate implements the [`poulpy_hal`] extension points unconditionally. The
//! higher layers are opt-in:
//!
//! - `enable-core`: implements the `poulpy-core` extension points, so
//!   `Module<FFT64Ref>` / `Module<NTT4x30Ref>` gain the scheme-level traits
//!   (`GLWEKeyswitch`, `Automorphism`, ...). Without it those traits do not
//!   resolve, and the failure reads as a missing impl rather than a missing
//!   feature. Required to use this crate as the reference side of a
//!   cross-backend comparison.
//! - `enable-ckks`: implies `enable-core` and adds the `poulpy-ckks` layer.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

#[cfg(feature = "enable-ckks")]
pub mod ckks_encoding;
#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-ckks")]
pub mod ckks_paco;
#[cfg(feature = "enable-ckks")]
pub mod ckks_ship;
#[cfg(feature = "enable-core")]
#[doc(hidden)]
pub mod core_impl;
pub mod fft64;
pub mod hal_defaults;
mod hal_impl;
pub mod ntt4x30;
pub mod reference;
pub mod table_cache;

#[cfg(test)]
mod tests;

pub use poulpy_hal::cast_mut;

pub mod api {
    pub use poulpy_hal::api::*;
}

pub mod layouts {
    pub use poulpy_hal::layouts::*;
}

pub mod source {
    pub use poulpy_hal::source::*;
}

pub use fft64::{FFT64Ref, FFT64ReimTable};
pub use ntt4x30::{NTT4x30Ref, NTT4x30RefHandle};
