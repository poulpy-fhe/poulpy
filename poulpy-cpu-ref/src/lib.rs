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
//! [`FFT64CIRef`] and [`NTT4x30CIRef`] serve the conjugate invariant ring. Each lives in its
//! own module ([`fft64_ci`], [`ntt4x30_ci`]) and forwards the ring-independent kernels to its
//! standard counterpart through the exported `forward_*!` macros.
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

mod backend_defaults;

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
pub mod fft64_ci;
mod forward;
pub mod hal_defaults;
mod hal_impl;
pub mod ntt4x30;
pub mod ntt4x30_ci;
mod sampling;
mod scalar_znx_fill;

pub mod capabilities;
pub mod reference;
pub mod table_cache;
pub mod test_suite;

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

pub use scalar_znx_fill::ScalarZnxFill;
pub mod ring;

pub use fft64::{FFT64Ref, FFT64ReimTable};
pub use ntt4x30::{NTT4x30Ref, NTT4x30RefHandle};

#[cfg(test)]
crate::conjugate_invariant_test_suite!(ci_fft64ref, crate::FFT64CIRef, crate::FFT64Ref);

#[cfg(test)]
crate::conjugate_invariant_test_suite!(ci_ntt4x30ref, crate::NTT4x30CIRef, crate::NTT4x30Ref);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(ci_core_fft64ref, crate::FFT64CIRef, crate::FFT64Ref);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(ci_core_ntt4x30ref, crate::NTT4x30CIRef, crate::NTT4x30Ref);

#[cfg(feature = "enable-ckks")]
mod ckks_comparison;

pub use fft64_ci::FFT64CIRef;
pub use ntt4x30_ci::{NTT4x30CIRef, NTT4x30CIRefHandle};

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
