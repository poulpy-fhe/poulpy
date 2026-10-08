//! Rayon-scheduled variants of the portable backends.
//!
//! They live in this crate because it already depends on `poulpy-cpu-portable`: the portable crate cannot
//! depend on the executor in return.

mod fft64;
#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

/// Rayon-scheduled variant of [`FFT64Portable`](poulpy_cpu_portable::FFT64Portable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64PortableRayon<R: Ring = Standard>(PhantomData<R>);

/// [`FFT64PortableRayon`] over the conjugate invariant ring.
pub type FFT64CIPortableRayon = FFT64PortableRayon<ConjugateInvariant>;
