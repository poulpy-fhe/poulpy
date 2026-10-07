//! Portable NTT4x30 CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT4x30Portable`], a backend implementation for [`poulpy_hal`] on the CRT of four
//! primes of about 30 bits. It runs on every CPU architecture, in plain scalar Rust.
//!
//! # Transform domain
//!
//! A transformed limb is four planes of `n` canonical `u32` residues, 16 bytes per coefficient.
//! Prepared operands (`SvpPPol`, `VmpPMat`, the convolution operands) hold the same residues, multiplied by `2^32`
//! or centered around zero where their products need it, in the order their kernels read them.
//!
//! The kernels are written as loops with a fixed stride over one prime at a time, the shape compilers turn into
//! vector instructions on the targets that have them. The q120 kernels of `crate::kernels::ntt4x30`, which the
//! SIMD backends build on, use a different layout and are not part of this backend's transform path.
//!
//! | Module          | Domain                                                              |
//! |-----------------|---------------------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, NTT tables                                |
//! | `packed`        | Modular arithmetic and inner products on packed residues            |
//! | `ntt32`         | Forward and inverse transform of one packed limb                    |
//! | `vec_znx_dft`   | Transform-domain vectors                                            |
//! | `svp`           | Scalar-vector product                                               |
//! | `vmp`           | Vector-matrix product                                               |
//! | `convolution`   | Bivariate convolution                                               |
//! | `hal_impl`      | Wiring of the above into the `poulpy_hal` extension points          |
//! | `znx`           | Single ring element (`Z[X]/(X^n+1)`) arithmetic                     |
//! | `vec_znx_big`   | Large-coefficient (`i128`) vectors, on the shared NTT4x30 defaults  |
//!
//! # Scalar types
//!
//! - `DftWord = CrtWord<Primes30, u32>`: four residues of a coefficient in the transform domain (16 bytes).
//! - `BigWord = i128`: coefficients in the large-integer (CRT-reconstructed) domain.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

mod convolution;
mod hal_impl;
mod module;
mod ntt32;
mod packed;
mod prim;
mod svp;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;
mod znx;

pub use module::NTT4x30PortableHandle;

use std::marker::PhantomData;

use poulpy_hal::layouts::{Ring, Standard};

/// Portable CPU backend using Q120 NTT arithmetic.
///
/// `NTT4x30Portable<R>` is a zero-sized marker type that selects the portable NTT4x30 CPU backend
/// over ring `R` (the standard ring by default, [`NTT4x30CIPortable`](crate::NTT4x30CIPortable) for the
/// conjugate invariant ring)
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **DftWord**: `CrtWord<Primes30, u32>`, NTT-domain coefficients stored as four canonical `u32` CRT residues.
/// - **BigWord**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes30` (four ~30-bit primes, Q ≈ 2^120).
/// - **NTT tables**: precomputed twiddle factors stored in the module handle
///   (`NTT4x30PortableHandle`), shared across all operations on the same module.
///
/// # Thread safety
///
/// `NTT4x30Portable` is `Send + Sync` (derived from being a zero-sized struct).
/// The `Module<NTT4x30Portable>` that holds the NTT tables is also `Send + Sync`, so modules can
/// be shared across threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30Portable<R: Ring = Standard>(PhantomData<R>);
