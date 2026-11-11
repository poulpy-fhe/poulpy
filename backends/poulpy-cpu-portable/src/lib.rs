#![feature(portable_simd, hint_prefetch, associated_type_defaults)]
#![allow(clippy::too_many_arguments)]
//! Portable CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two backends for [`poulpy_hal`]:
//!
//! - [`FFT64Portable`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT4x30Portable`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt4x30`] module.
//!
//! Both are portable across all CPU architectures, in plain Rust (`std::simd` for normalization), and
//! their scalar kernels, in [`kernels`], back the SIMD backends too.
//!
//! Both serve the standard and the conjugate-invariant ring: the ring is a parameter of the operands.
#![cfg_attr(
    feature = "enable-core",
    doc = r"
# Rings

The Galois trace serves both rings:
```
use poulpy_core::GLWETrace;
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::layouts::{ConjugateInvariant, Module};
fn trace<M: GLWETrace<FFT64Portable, ConjugateInvariant>>(_: &M) {}
fn check(module: &Module<FFT64Portable>) { trace(module); }
```
Monomial rotation takes standard-ring operands:
```
use poulpy_core::{GLWERotate, layouts::GLWE};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::{AlignedBuf, layouts::Module};
fn rotate(module: &Module<FFT64Portable>, res: &mut GLWE<AlignedBuf, i64>, a: &GLWE<AlignedBuf, i64>) {
    module.glwe_rotate(1, res, a);
}
```
```compile_fail,E0277
use poulpy_core::{GLWERotate, layouts::GLWE};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::{AlignedBuf, layouts::{ConjugateInvariant, Module}};
fn rotate(
    module: &Module<FFT64Portable>,
    res: &mut GLWE<AlignedBuf, i64, ConjugateInvariant>,
    a: &GLWE<AlignedBuf, i64, ConjugateInvariant>,
) {
    module.glwe_rotate(1, res, a);
}
```
Prepared keys retain their ring:
```
use poulpy_core::layouts::{GLWETensorKeyPrepared, GetTensorKey};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{AlignedBuf, layouts::ConjugateInvariant};
fn accepts_ci(_: &impl GetTensorKey<NTT4x30Portable, ConjugateInvariant>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30Portable, ConjugateInvariant>) { accepts_ci(key); }
```
```compile_fail,E0277
use poulpy_core::layouts::{GLWETensorKeyPrepared, GetTensorKey};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{AlignedBuf, layouts::ConjugateInvariant};
fn accepts_ci(_: &impl GetTensorKey<NTT4x30Portable, ConjugateInvariant>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30Portable>) { accepts_ci(key); }
```"
)]
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
//! - `enable-blas`: implies `enable-ckks` and wires the enabled backends into `poulpy-blas`.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

mod backend_defaults;

#[cfg(feature = "enable-blas")]
mod blas_impl;
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
    crate::FFT64Portable,
    reference = poulpy_cpu_oracle::FFT64Oracle,
    large_radix_reference = poulpy_cpu_oracle::NTT4x30Oracle
);

#[cfg(test)]
crate::conjugate_invariant_test_suite!(
    ci_ntt4x30portable,
    crate::NTT4x30Portable,
    reference = poulpy_cpu_oracle::FFT64Oracle,
    large_radix_reference = poulpy_cpu_oracle::NTT4x30Oracle
);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(
    ci_core_fft64portable,
    crate::FFT64Portable,
    reference = poulpy_cpu_oracle::FFT64Oracle
);

#[cfg(all(test, feature = "enable-core"))]
crate::conjugate_invariant_core_test_suite!(
    ci_core_ntt4x30portable,
    crate::NTT4x30Portable,
    reference = poulpy_cpu_oracle::FFT64Oracle
);

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_fft64portable,
    crate::FFT64Portable,
    poulpy_ckks::test_suite::BASE19_PARAMS_F64
);

#[cfg(all(test, feature = "enable-ckks"))]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_ntt4x30portable,
    crate::NTT4x30Portable,
    poulpy_ckks::test_suite::BASE52_PARAMS_F64
);

#[cfg(feature = "enable-ckks")]
mod ckks_comparison;

/// The names of this crate's backends before it was renamed from `poulpy-cpu-ref`.
#[deprecated(note = "renamed to `FFT64Portable`")]
pub type FFT64Ref = FFT64Portable;
#[deprecated(note = "renamed to `NTT4x30Portable`")]
pub type NTT4x30Ref = NTT4x30Portable;
/// Former CKKS encoding transform, now [`EncodingFFTTable`](ckks_encoding::EncodingFFTTable).
#[cfg(feature = "enable-ckks")]
#[deprecated(note = "use `ckks_encoding::EncodingFFTTable`, which encodes byte identically on every CPU backend")]
pub type FFT64ReimTable<F> = ckks_encoding::EncodingFFTTable<F>;

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
