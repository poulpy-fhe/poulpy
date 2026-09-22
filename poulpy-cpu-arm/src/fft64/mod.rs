//! NEON-accelerated FFT64 CPU backend.

use poulpy_cpu_ref::ring::CpuRing;

mod module;
#[cfg(feature = "enable-rayon")]
mod rayon;
mod reim;
mod znx;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;
pub use reim::{FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};

/// NEON-accelerated CPU backend for Poulpy HAL.
/// `DftWord = f64`, `BigWord = i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64NeonBackend<R: CpuRing = poulpy_cpu_ref::ring::Standard>(std::marker::PhantomData<R>);

/// Standard negacyclic backend.
pub type FFT64Neon = FFT64NeonBackend<poulpy_cpu_ref::ring::Standard>;
/// Conjugate-invariant backend.
pub type FFT64CINeon = FFT64NeonBackend<poulpy_cpu_ref::ring::ConjugateInvariant>;

/// Rayon-scheduled variant of [`FFT64Neon`].
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64NeonRayonBackend<R: CpuRing = poulpy_cpu_ref::ring::Standard>(std::marker::PhantomData<R>);

/// Standard negacyclic Rayon backend.
#[cfg(feature = "enable-rayon")]
pub type FFT64NeonRayon = FFT64NeonRayonBackend<poulpy_cpu_ref::ring::Standard>;
/// Conjugate-invariant Rayon backend.
#[cfg(feature = "enable-rayon")]
pub type FFT64CINeonRayon = FFT64NeonRayonBackend<poulpy_cpu_ref::ring::ConjugateInvariant>;
