//! AVX512-IFMA accelerated NTT CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT3x42Ifma`], an AVX512-IFMA accelerated backend implementation for
//! [`poulpy_hal`] that uses IFMA NTT arithmetic (CRT over three ~42-bit primes). The
//! scalar reference for these kernels lives in the [`reference`] submodule.
//!
//! # Current acceleration status
//!
//! | Domain | Status |
//! |-|-|
//! | Coefficient-domain (`Znx*`) | AVX-512F (reuses `crate::znx_avx512`) |
//! | NTT forward/inverse | AVX512-IFMA (`kernels` module) |
//! | mat_vec BBC product (SVP/VMP hot path) | AVX512-IFMA (`mat_vec_ifma` module) |
//! | VecZnxBig add/sub/negate | shared `i128` helpers wired through the HAL implementation |
//! | VecZnxBig normalization | shared `i128` normalization helpers wired through the HAL implementation |
//!
//! # Scalar types
//!
//! - `DftWord = Q126Scalar` — an identity marker; DFT storage packs three
//!   42-bit prime residues into two `u64` words per coefficient.
//! - `BigWord  = i128` — CRT-reconstructed large coefficients.

pub(crate) mod bbc_meta;
pub(crate) mod convolution;
mod execution;
pub(crate) mod kernels;
pub(crate) mod mat_vec_ifma;
pub(crate) mod module;
mod prim;
pub(crate) mod primes;
#[cfg(feature = "enable-rayon")]
pub(crate) mod rayon;
#[cfg(feature = "enable-rayon")]
pub(crate) use rayon::vmp_apply_digits_strided_known_zero_prefix;
pub(crate) mod reference;
pub(crate) mod svp;
pub(crate) mod tables;
pub(crate) mod traits;
pub(crate) mod types;
mod vec_znx_big;
pub(crate) mod vec_znx_dft;
pub(crate) mod vmp;
mod znx;

#[cfg(test)]
mod tests;

/// AVX512-IFMA accelerated NTT CPU backend for Poulpy HAL.
///
/// `NTT3x42Ifma` is a zero-sized marker type that selects the AVX512-IFMA accelerated NTT backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **DftWord**: `Q126Scalar` — an identity marker for a packed two-word
///   representation of three 42-bit prime residues.
/// - **BigWord**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes42` (three ~42-bit primes, Q ≈ 2^126).
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::api::ModuleNew::new) verifies that
/// the CPU supports AVX512-F, AVX512-IFMA, and AVX512-VL. If a required
/// feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTT3x42Ifma` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT3x42Ifma;

/// Rayon-parallel AVX512-IFMA backend.
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT3x42IfmaRayon;

#[cfg(feature = "enable-rayon")]
pub type NTT3x42IfmaRayonExecutor = poulpy_cpu_rayon::RayonTaskExecutor;

#[cfg(feature = "enable-rayon")]
poulpy_hal::impl_backend_from!(NTT3x42IfmaRayon, NTT3x42Ifma, NTT3x42IfmaRayonExecutor);
