//! Rayon-scheduled variants of the portable backends.
//!
//! They live in this crate because it already depends on `poulpy-cpu-portable`: the portable crate cannot
//! depend on the executor in return.

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-core")]
mod core_impl;
mod fft64;
#[cfg(feature = "enable-mhe")]
mod mhe_impl;
mod ntt4x30;
#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

/// Rayon-scheduled variant of [`FFT64Portable`](poulpy_cpu_portable::FFT64Portable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64PortableRayon<R: Ring = Standard>(PhantomData<R>);

/// [`FFT64PortableRayon`] over the conjugate invariant ring.
pub type FFT64CIPortableRayon = FFT64PortableRayon<ConjugateInvariant>;

/// Rayon-scheduled variant of [`NTT4x30Portable`](poulpy_cpu_portable::NTT4x30Portable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30PortableRayon<R: Ring = Standard>(PhantomData<R>);

/// [`NTT4x30PortableRayon`] over the conjugate invariant ring.
pub type NTT4x30CIPortableRayon = NTT4x30PortableRayon<ConjugateInvariant>;
