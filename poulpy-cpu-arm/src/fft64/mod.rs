//! NEON-accelerated FFT64 CPU backend.

mod conjugate_invariant;
mod module;
#[cfg(feature = "enable-rayon")]
mod rayon;
mod reim;
mod standard;
mod znx;

use std::marker::PhantomData;

use poulpy_hal::layouts::{Ring, Standard};

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_portable::kernels::fft64::module::FFTModuleHandle;
pub use reim::{FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};

/// NEON-accelerated CPU backend for Poulpy HAL.
/// `DftWord = f64`, `BigWord = i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64Neon<R: Ring = Standard>(PhantomData<R>);

/// Rayon-scheduled variant of [`FFT64Neon`].
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64NeonRayon<R: Ring = Standard>(PhantomData<R>);
