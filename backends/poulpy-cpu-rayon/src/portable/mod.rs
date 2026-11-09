//! Rayon-scheduled variants of the portable backends.
//!
//! They live in this crate because it already depends on `poulpy-cpu-portable`: the portable crate cannot
//! depend on the executor in return.

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_impl;
#[cfg(feature = "enable-blas")]
mod blas_impl;
#[cfg(all(test, feature = "enable-blas"))]
mod blas_tests;
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

/// Rayon-scheduled variant of [`FFT64Portable`](poulpy_cpu_portable::FFT64Portable), on both rings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct FFT64PortableRayon;

/// Rayon-scheduled variant of [`NTT4x30Portable`](poulpy_cpu_portable::NTT4x30Portable), on both rings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NTT4x30PortableRayon;
