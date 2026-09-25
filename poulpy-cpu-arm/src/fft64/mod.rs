//! NEON-accelerated FFT64 CPU backend.

mod encoding;
mod module;
#[cfg(feature = "enable-rayon")]
mod rayon;
pub(crate) mod reim;
mod znx;

#[cfg(test)]
mod tests;

pub use encoding::{FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};
#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;

/// NEON-accelerated CPU backend for Poulpy HAL.
/// `DftWord = f64`, `BigWord = i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64Neon;

/// Rayon-scheduled variant of [`FFT64Neon`].
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64NeonRayon;
