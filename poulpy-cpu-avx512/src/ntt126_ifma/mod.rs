//! AVX512-IFMA accelerated NTT CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT126Ifma`], an AVX512-IFMA accelerated backend implementation for
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
//! - `ScalarPrep = Q120bScalar` — shared 4-lane prep scalar (three active residues plus padding).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

pub(crate) mod bbc_meta;
pub(crate) mod convolution;
pub(crate) mod kernels;
pub(crate) mod mat_vec_ifma;
pub(crate) mod module;
mod prim;
pub(crate) mod primes;
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
/// `NTT126Ifma` is a zero-sized marker type that selects the AVX512-IFMA accelerated NTT backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — shared 4-lane prep scalar with three CRT residues plus one padding lane.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes42` (three ~42-bit primes, Q ≈ 2^126).
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::api::ModuleNew::new) verifies that
/// the CPU supports AVX512-F, AVX512-IFMA, AVX512-VL, BMI2, and ADX. If a required
/// feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTT126Ifma` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy)]
pub struct NTT126Ifma;
