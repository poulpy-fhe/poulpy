//! NEON-accelerated FFT64 CPU backend.
//!
//! All HAL extension points (`Znx*`, `ReimArith`, `Reim4*`, `I64Ops`,
//! `ReimFFTExecute`, …) are routed through the NEON kernels in
//! [`crate::neon`] on `target_arch = "aarch64"` and through `poulpy-cpu-ref`
//! otherwise.

mod module;
mod reim;
mod znx;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use poulpy_cpu_ref::reference::fft64::module::FFTModuleHandle;
pub use reim::{FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};

/// NEON-accelerated CPU backend for Poulpy HAL.
///
/// `FFT64Neon` is a zero-sized marker type that selects the AArch64 NEON CPU
/// backend when used as the type parameter `B` in
/// [`Module<B>`](poulpy_hal::layouts::Module). It implements all open extension
/// point (OEP) traits from `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `f64` — DFT-domain coefficients are 64-bit IEEE 754 floats.
/// - **ScalarBig**: `i64` — large-coefficient ring elements use 64-bit signed integers.
/// - **FFT tables**: precomputed twiddle factors stored in the module handle
///   ([`module::FFT64NeonHandle`]), shared across all operations on the same module.
#[derive(Debug, Clone, Copy)]
pub struct FFT64Neon;
