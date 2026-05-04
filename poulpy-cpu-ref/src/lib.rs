//! Reference (portable) CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two reference implementations for [`poulpy_hal`]:
//!
//! - [`FFT64Ref`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT120Ref`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt120`] module.
//!
//! Both are canonical reference implementations: portable across all CPU architectures,
//! prioritising correctness and debuggability over throughput.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

pub mod fft64;
pub mod hal_defaults;
mod hal_impl;
pub mod ntt120;
pub mod reference;

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

pub use fft64::FFT64Ref;
pub use ntt120::{NTT120Ref, NTT120RefHandle};

use poulpy_core::oep::CoreImpl;
unsafe impl CoreImpl<FFT64Ref> for FFT64Ref {
    poulpy_core::impl_core_default_methods!(FFT64Ref);
}

unsafe impl CoreImpl<NTT120Ref> for NTT120Ref {
    poulpy_core::impl_core_default_methods!(NTT120Ref);
}
