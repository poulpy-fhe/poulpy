//! AVX-512F accelerated NTT120 CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT120Avx512`], an AVX-512F accelerated backend implementation for
//! [`poulpy_hal`] that uses Q120 NTT arithmetic (CRT over four ~30-bit primes). It mirrors
//! the structure of the scalar [`poulpy_cpu_ref::NTT120Ref`] backend, with AVX-512F
//! accelerated kernels substituted where available.
//!
//! # Layout strategy
//!
//! Each Q120bScalar coefficient occupies a 256-bit half (4 × u64, one CRT residue per lane).
//! The AVX-512F widening packs **two coefficients per 512-bit register**:
//!
//! ```text
//! __m512i = [r0_A, r1_A, r2_A, r3_A,  r0_B, r1_B, r2_B, r3_B]
//!           └──── coefficient A ────┘└──── coefficient B ────┘
//! ```
//!
//! Per-prime constants (`Q_SHIFTED`, etc.) and twiddle entries (4 × u64) are broadcast to
//! both halves with `_mm512_broadcast_i64x4`. Both halves perform the same operation on
//! independent coefficients in parallel.
//!
//! # Current acceleration status
//!
//! | Domain | Status |
//! |-|-|
//! | Coefficient-domain (`Znx*`) | AVX-512F (reuses `crate::znx_avx512`) |
//! | NTT lazy add/sub/negate (`prim`) | AVX-512F, 2-coefficient pair-pack |
//! | NTT level-0 element-wise twiddle (`ntt::ntt_iter_first*`) | AVX-512F, 2-coefficient pair-pack |
//! | NTT butterfly levels (`ntt::ntt_iter*`, `intt_iter*`) | AVX-512F when `halfnn ≥ 4` (within-block i,i+1 pair-pack); 256-bit fallback for `halfnn ∈ {1, 2}` |
//! | Domain conversions (`arithmetic_avx512::b_from_znx64*`, `c_from_b`, `b_to_znx128`) | AVX-512F, 2-coefficient pair-pack |
//! | BBB inner product (`arithmetic_avx512::vec_mat1col_product_bbb`) | AVX-512F, 2-element pair-pack with half-fold |
//! | BBC mat-vec (`mat_vec_avx512::vec_mat1col_product_bbc`, `_x2_bbc`, `_2cols_x2_bbc`, `_blkpair_bbc_pm`) | AVX-512F, 2-element pair-pack with half-fold |
//! | Pack helpers (`arithmetic_avx512::pack_*_1blk_x2*`, `pairwise_pack_*_1blk_x2*`) | AVX-512F (pair-pack the two q120b's per row) |
//! | VecZnxBig i128 ops + normalization (`crate::vec_znx_big_avx512`) | AVX-512F, 4-i128/512-bit (mask-based borrow) |
//!
//! Block-order tight inner stages (`nn = 2, 4`) cannot pair-pack along `i` (≤ 1 twiddled
//! butterfly per block). `nn = 2` uses a cross-block pair-pack kernel; `nn = 4` currently
//! falls back to 256-bit and is a candidate for cross-block pair-packing.
//!
//! # Scalar types
//!
//! - `ScalarPrep = Q120bScalar` — NTT-domain coefficients (4 × u64, 32 bytes/coeff).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

pub(crate) mod arithmetic_avx512;
pub(crate) mod convolution;
pub(crate) mod mat_vec_avx512;
mod module;
pub(crate) mod ntt;
mod prim;
mod vec_znx_big;
pub(crate) mod vec_znx_dft_consume;
pub(crate) mod vmp;
mod znx;

/// AVX-512F-accelerated NTT120 CPU backend for Poulpy HAL.
///
/// `NTT120Avx512` is a zero-sized marker type that selects the AVX-512F NTT120 backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — NTT-domain coefficients stored as 4 × u64 CRT residues.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes30` (four ~30-bit primes, Q ≈ 2^120).
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::api::ModuleNew::new) verifies that
/// the CPU supports AVX-512F. If the feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTT120Avx512` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy)]
pub struct NTT120Avx512;

#[cfg(test)]
mod tests;
