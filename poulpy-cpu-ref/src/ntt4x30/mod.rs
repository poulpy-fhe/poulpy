//! Reference NTT4x30 CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides [`NTT4x30Ref`], a backend implementation for [`poulpy_hal`] that uses
//! scalar Q120 NTT arithmetic (Chinese Remainder Theorem over four ~30-bit primes). It is the
//! reference NTT4x30 implementation: portable across all CPU architectures, prioritising
//! correctness and debuggability over throughput.
//!
//! # Architecture
//!
//! `poulpy-hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for the [`NTT4x30Ref`] backend by delegating to the reference
//! functions provided by `crate::reference::ntt4x30`.
//!
//! The internal modules are organised by operation domain:
//!
//! | Module          | Domain                                                         |
//! |-----------------|----------------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, NTT table management                 |
//! | `scratch`       | Temporary memory allocation and arena-style sub-allocation, now provided by shared `poulpy-hal` portable defaults |
//! | `znx`           | Single ring element (`Z[X]/(X^n+1)`) arithmetic               |
//! | `vec_znx`       | Vectors of ring elements (limb decomposition), now provided by shared `poulpy-hal` portable defaults |
//! | `vec_znx_big`   | Large-coefficient (i128) ring element vectors, now provided by shared `poulpy-hal` NTT4x30 defaults |
//! | `vec_znx_dft`   | NTT-domain ring element vectors, now provided by shared `poulpy-hal` NTT4x30 defaults |
//! | `svp`           | Scalar-vector product in NTT domain, now provided by shared `poulpy-hal` NTT4x30 defaults |
//!
//! # Scalar types
//!
//! For the `NTT4x30Ref` backend:
//!
//! - `ScalarPrep = Q120bScalar`: coefficients in the NTT / frequency domain (32 bytes = 4 × u64).
//! - `ScalarBig  = i128`: coefficients in the large-integer (CRT-reconstructed) domain.
//!
//! # Usage
//!
//! This crate exports a single public type, [`NTT4x30Ref`], which is used as a type parameter
//! to the HAL generic types. All functionality is accessed through the trait methods defined
//! in `poulpy_hal::api`.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

mod module;
mod prim;
mod vec_znx_big;
mod znx;

pub use module::NTT4x30RefHandle;

/// Reference (portable) CPU backend using Q120 NTT arithmetic.
///
/// `NTT4x30Ref` is a zero-sized marker type that selects the reference NTT4x30 CPU backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep` by delegating to the portable reference functions in
/// `crate::reference::ntt4x30`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — NTT-domain coefficients stored as 4 × u64 CRT residues.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes30` (four ~30-bit primes, Q ≈ 2^120).
/// - **NTT tables**: precomputed twiddle factors stored in the module handle
///   (`NTT4x30RefHandle`), shared across all operations on the same module.
///
/// # Thread safety
///
/// `NTT4x30Ref` is `Send + Sync` (derived from being a zero-sized, field-less struct).
/// The `Module<NTT4x30Ref>` that holds the NTT tables is also `Send + Sync`, so modules can
/// be shared across threads.
#[derive(Debug, Clone, Copy)]
pub struct NTT4x30Ref;
