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
//! Both are generic over the ring, standard by default: [`FFT64CIRef`] and [`NTT4x30CIRef`]
//! are their conjugate invariant instantiations.
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
pub mod hal_defaults;
mod hal_impl;
pub mod ntt4x30;
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

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_fft64ref,
    crate::FFT64CIRef,
    poulpy_ckks::test_suite::BASE19_PARAMS_F64
);

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_ntt4x30ref,
    crate::NTT4x30CIRef,
    poulpy_ckks::test_suite::BASE52_PARAMS_F64
);

#[cfg(feature = "enable-ckks")]
mod ckks_comparison;

/// [`FFT64Ref`] over the conjugate invariant ring.
#[cfg_attr(
    feature = "enable-core",
    doc = r"
The Galois trace is standard-only:
```
use poulpy_core::GLWETrace;
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::Module;
fn trace<M: GLWETrace<FFT64Ref>>(_: &M) {}
fn check(module: &Module<FFT64Ref>) { trace(module); }
```
```compile_fail
use poulpy_core::GLWETrace;
use poulpy_cpu_ref::FFT64CIRef;
use poulpy_hal::layouts::Module;
fn trace<M: GLWETrace<FFT64CIRef>>(_: &M) {}
fn check(module: &Module<FFT64CIRef>) { trace(module); }
```"
)]
pub type FFT64CIRef = FFT64Ref<poulpy_hal::layouts::ConjugateInvariant>;

/// [`NTT4x30Ref`] over the conjugate invariant ring.
#[cfg_attr(
    feature = "enable-core",
    doc = r"
Prepared keys retain their backend type:
```
use poulpy_cpu_ref::NTT4x30CIRef;
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIRef>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30CIRef>) { accepts_ci(key); }
```
```compile_fail
use poulpy_cpu_ref::{NTT4x30CIRef, NTT4x30Ref};
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIRef>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30Ref>) { accepts_ci(key); }
```"
)]
pub type NTT4x30CIRef = NTT4x30Ref<poulpy_hal::layouts::ConjugateInvariant>;

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
