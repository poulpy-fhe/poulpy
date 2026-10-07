//! Portable (scalar f64 FFT) backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`FFT64Portable`], a backend implementation for [`poulpy_hal`] that uses
//! scalar `f64` FFT arithmetic across CPU architectures. Independent correctness
//! oracles are provided by `poulpy-cpu-oracle`.
//!
//! # Architecture
//!
//! `poulpy-hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This module
//! implements every OEP trait for the [`FFT64Portable`] backend by delegating to the portable
//! functions provided by `crate::kernels::fft64`.
//!
//! The internal modules are organised by operation domain:
//!
//! | Module          | Domain                                                    |
//! |-----------------|-----------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, FFT table management            |
//! | `scratch`       | Temporary memory allocation and arena-style sub-allocation, now provided by shared `poulpy-hal` portable defaults |
//! | `znx`           | Single ring element (`Z[X]/(X^n+1)`) arithmetic           |
//! | `vec_znx`       | Vectors of ring elements (limb decomposition), now provided by shared `poulpy-hal` portable defaults |
//! | `vec_znx_big`   | Large-coefficient (multi-word) ring element vectors, now provided by shared `poulpy-hal` FFT64 defaults |
//! | `vec_znx_dft`   | Fourier-domain ring element vectors, now provided by shared `poulpy-hal` FFT64 defaults |
//! | `reim`          | Real/imaginary interleaved FFT primitives                 |
//! | `svp`           | Scalar-vector product in frequency domain, now provided by shared `poulpy-hal` FFT64 defaults |
//!
//! # Scalar types
//!
//! - `DftWord = f64`: coefficients in the DFT / frequency domain.
//! - `BigWord  = i64`: coefficients in the large-integer (multi-word) domain.
//!   meaning each coefficient occupies exactly one scalar word.

mod module;
pub(crate) mod reim;
mod znx;

use std::marker::PhantomData;

use poulpy_hal::layouts::{Ring, Standard};

pub use crate::kernels::fft64::module::FFTModuleHandle;

/// Portable CPU backend using f64 FFT.
///
/// `FFT64Portable<R>` is a zero-sized marker type that selects the portable CPU backend
/// over ring `R` (the standard ring by default, [`FFT64CIPortable`](crate::FFT64CIPortable) for the
/// conjugate invariant ring)
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep` by delegating to the portable functions in
/// `crate::kernels::fft64`.
///
/// # Backend characteristics
///
/// - **DftWord**: `f64` — DFT-domain coefficients are 64-bit IEEE 754 floats.
/// - **BigWord**: `i64` — large-coefficient ring elements use 64-bit signed integers.
/// - **FFT tables**: precomputed twiddle factors stored in the module handle
///   (`FFT64PortableHandle`), shared across all operations on the same module.
///
/// # Thread safety
///
/// `FFT64Portable` is `Send + Sync` (derived from being a zero-sized struct).
/// The `Module<FFT64Portable>` that holds the FFT tables is also `Send + Sync`, so modules can
/// be shared across threads. Individual operations require exclusive (`&mut`) access to their
/// output buffers and scratch space, preventing data races at the API level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64Portable<R: Ring = Standard>(PhantomData<R>);
