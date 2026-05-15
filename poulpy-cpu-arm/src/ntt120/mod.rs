//! NEON-accelerated NTT120 CPU backend.
//!
//! This module provides [`NTT120Neon`], a NEON-accelerated backend implementation
//! for [`poulpy_hal`] that uses Q120 NTT arithmetic (CRT over four ~30-bit primes).
//! It mirrors the structure of the scalar [`poulpy_cpu_ref::NTT120Ref`] backend,
//! with NEON-accelerated kernels substituted on AArch64.
//!
//! All HAL extension points (`Znx*`, `Ntt*`, `NttDFTExecute`, conversions,
//! packs, mat-vec products, `I128BigOps`, `I128NormalizeOps`,
//! `vec_znx_idft_apply_consume`, …) are routed through the NEON kernels in
//! [`crate::neon`] on `target_arch = "aarch64"` and through `poulpy-cpu-ref`
//! otherwise.
//!
//! # Scalar types
//!
//! - `ScalarPrep = Q120bScalar` — NTT-domain coefficients (4 × u64, 32 bytes/coeff).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

mod module;
mod prim;
mod vec_znx_big;
mod znx;

#[cfg(test)]
mod tests;

/// NEON-accelerated NTT120 CPU backend for Poulpy HAL.
///
/// `NTT120Neon` is a zero-sized marker type that selects the AArch64 NEON
/// NTT120 backend when used as the type parameter `B` in
/// [`Module<B>`](poulpy_hal::layouts::Module). It implements all open extension
/// point (OEP) traits from `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — NTT-domain coefficients stored as 4 × u64 CRT residues.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes30` (four ~30-bit primes, Q ≈ 2^120).
#[derive(Debug, Clone, Copy)]
pub struct NTT120Neon;
