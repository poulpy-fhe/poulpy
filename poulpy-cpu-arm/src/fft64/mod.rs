//! NEON-accelerated FFT64 CPU backend.

mod module;
mod reim;
mod znx;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;
pub use reim::{FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};

/// NEON-accelerated CPU backend for Poulpy HAL.
/// `DftWord = f64`, `BigWord = i64`.
#[derive(Debug, Clone, Copy)]
pub struct FFT64Neon;
