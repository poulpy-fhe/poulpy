#![allow(clippy::too_many_arguments)]

//! Portable CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two backends for [`poulpy_hal`]:
//!
//! - [`FFT64Portable`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT4x30Portable`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt4x30`] module.
//!
//! Both are portable across all CPU architectures, in plain scalar Rust, and
//! their scalar kernels, in [`kernels`], back the SIMD backends too.
//!
//! Both are generic over the ring, standard by default: [`FFT64CIPortable`] and [`NTT4x30CIPortable`]
//! are their conjugate invariant instantiations.
//!
//! # Features
//!
//! The crate implements the [`poulpy_hal`] extension points unconditionally. The
//! higher layers are opt-in:
//!
//! - `enable-core`: implements the `poulpy-core` extension points, so
//!   `Module<FFT64Portable>` / `Module<NTT4x30Portable>` gain the scheme-level traits
//!   (`GLWEKeyswitch`, `Automorphism`, ...). Without it those traits do not
//!   resolve, and the failure reads as a missing impl rather than a missing
//!   feature. Required to use this crate on either side of a cross-backend
//!   comparison.
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
#[cfg(feature = "enable-mhe")]
mod mhe_impl;
pub mod ntt4x30;
mod sampling;
mod scalar_znx_fill;

pub mod capabilities;
pub mod kernels;
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

pub use fft64::FFT64Portable;
pub use ntt4x30::{NTT4x30Portable, NTT4x30PortableHandle};

#[cfg(test)]
crate::conjugate_invariant_test_suite!(
    ci_fft64portable,
    crate::FFT64CIPortable,
    crate::FFT64Portable,
    reference = poulpy_cpu_oracle::FFT64CIOracle,
    large_radix_reference = poulpy_cpu_oracle::NTT4x30CIOracle
);

#[cfg(test)]
crate::conjugate_invariant_test_suite!(
    ci_ntt4x30portable,
    crate::NTT4x30CIPortable,
    crate::NTT4x30Portable,
    reference = poulpy_cpu_oracle::FFT64CIOracle,
    large_radix_reference = poulpy_cpu_oracle::NTT4x30CIOracle
);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(
    ci_core_fft64portable,
    crate::FFT64CIPortable,
    crate::FFT64Portable,
    reference = poulpy_cpu_oracle::FFT64CIOracle
);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(
    ci_core_ntt4x30portable,
    crate::NTT4x30CIPortable,
    crate::NTT4x30Portable,
    reference = poulpy_cpu_oracle::FFT64CIOracle
);

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_fft64portable,
    crate::FFT64CIPortable,
    crate::FFT64Portable,
    poulpy_ckks::test_suite::BASE19_PARAMS_F64
);

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_ntt4x30portable,
    crate::NTT4x30CIPortable,
    crate::NTT4x30Portable,
    poulpy_ckks::test_suite::BASE52_PARAMS_F64
);

#[cfg(feature = "enable-ckks")]
mod ckks_comparison;

/// [`FFT64Portable`] over the conjugate invariant ring.
#[cfg_attr(
    feature = "enable-core",
    doc = r"
The Galois trace is standard-only:
```
use poulpy_core::GLWETrace;
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::layouts::Module;
fn trace<M: GLWETrace<FFT64Portable>>(_: &M) {}
fn check(module: &Module<FFT64Portable>) { trace(module); }
```
```compile_fail
use poulpy_core::GLWETrace;
use poulpy_cpu_portable::FFT64CIPortable;
use poulpy_hal::layouts::Module;
fn trace<M: GLWETrace<FFT64CIPortable>>(_: &M) {}
fn check(module: &Module<FFT64CIPortable>) { trace(module); }
```
So are the embedding and trace between the two rings:
```
use poulpy_core::{GLWECITrace, GLWECIEmbed};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::layouts::Module;
fn maps<M: GLWECIEmbed<FFT64Portable> + GLWECITrace<FFT64Portable>>(_: &M) {}
fn check(module: &Module<FFT64Portable>) { maps(module); }
```
```compile_fail
use poulpy_core::GLWECIEmbed;
use poulpy_cpu_portable::FFT64CIPortable;
use poulpy_hal::layouts::Module;
fn embed<M: GLWECIEmbed<FFT64CIPortable>>(_: &M) {}
fn check(module: &Module<FFT64CIPortable>) { embed(module); }
```"
)]
pub type FFT64CIPortable = FFT64Portable<poulpy_hal::layouts::ConjugateInvariant>;

/// [`NTT4x30Portable`] over the conjugate invariant ring.
#[cfg_attr(
    feature = "enable-core",
    doc = r"
Prepared keys retain their backend type:
```
use poulpy_cpu_portable::NTT4x30CIPortable;
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIPortable>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30CIPortable>) { accepts_ci(key); }
```
```compile_fail
use poulpy_cpu_portable::{NTT4x30CIPortable, NTT4x30Portable};
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIPortable>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30Portable>) { accepts_ci(key); }
```"
)]
pub type NTT4x30CIPortable = NTT4x30Portable<poulpy_hal::layouts::ConjugateInvariant>;

/// The names of this crate's backends before it was renamed from `poulpy-cpu-ref`.
#[deprecated(note = "renamed to `FFT64Portable`")]
pub type FFT64Ref<R = poulpy_hal::layouts::Standard> = FFT64Portable<R>;
#[deprecated(note = "renamed to `NTT4x30Portable`")]
pub type NTT4x30Ref<R = poulpy_hal::layouts::Standard> = NTT4x30Portable<R>;
#[deprecated(note = "renamed to `FFT64CIPortable`")]
pub type FFT64CIRef = FFT64CIPortable;
#[deprecated(note = "renamed to `NTT4x30CIPortable`")]
pub type NTT4x30CIRef = NTT4x30CIPortable;
/// Former CKKS encoding transform, now [`EncodingFFTTable`](ckks_encoding::EncodingFFTTable).
#[cfg(feature = "enable-ckks")]
#[deprecated(note = "use `ckks_encoding::EncodingFFTTable`, which encodes byte identically on every CPU backend")]
pub type FFT64ReimTable<F> = ckks_encoding::EncodingFFTTable<F>;

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
