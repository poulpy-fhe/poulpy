//! AVX-512 / AVX-512-IFMA accelerated CPU backends for the Poulpy lattice cryptography library.
//!
//! This crate provides six backend implementations for [`poulpy_hal`]:
//!
//! - `FFT64Avx512`: f64 FFT backend, gated on `enable-avx512f`.
//! - `FFT64Avx512Rayon`: Rayon-scheduled FFT backend, gated on `enable-rayon`.
//! - `NTT4x30Avx512`: Q120 NTT backend over four ~30-bit CRT primes, gated on `enable-avx512f`.
//! - `NTT4x30Avx512Rayon`: Rayon-scheduled Q120 NTT backend, gated on `enable-rayon`.
//! - `NTT3x42Ifma`: Q126 NTT backend over three ~42-bit CRT primes, gated on `enable-ifma`.
//! - `NTT3x42IfmaRayon`: parallel IFMA backend, gated on `enable-ifma` and `enable-rayon`.
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the
//! [`Backend`](poulpy_hal::layouts::Backend) trait and open extension point
//! (OEP) traits in [`poulpy_hal::oep`]. This crate implements those extension
//! points with AVX-512F, AVX-512-IFMA, AVX2/FMA, and scalar/reference fallback
//! paths depending on the backend and operation family.
//!
//! The internal modules are organized by operation domain:
//!
//! | Module             | Domain                                                     |
//! |--------------------|------------------------------------------------------------|
//! | `fft64`            | `FFT64Avx512` backend and REIM FFT table wrappers          |
//! | `znx_avx512`       | AVX-512F single ring element arithmetic                    |
//! | `ntt4x30_avx512`    | `NTT4x30Avx512` NTT, VMP, convolution, and DFT kernels      |
//! | `ntt3x42_ifma`      | `NTT3x42Ifma` IFMA NTT, VMP, SVP, convolution, and DFT code |
//! | `hal_impl`         | HAL OEP implementations and default wiring                 |
//! | `vec_znx_big_avx512` | AVX-512F i128 accumulator helpers                        |
//!
//! # Scalar types
//!
//! - `FFT64Avx512`: `DftWord = f64`, `BigWord = i64`.
//! - `NTT4x30Avx512`: `DftWord = Q120bScalar`, `BigWord = i128`.
//! - `NTT3x42Ifma`: `DftWord = Q126Scalar`, `BigWord = i128`.
//!
//! # CPU requirements
//!
//! `FFT64Avx512` and `NTT4x30Avx512` require x86-64 with AVX-512F. The FFT64
//! backend also uses AVX2 and FMA kernels and checks those features at module
//! construction.
//!
//! `NTT3x42Ifma` additionally requires AVX-512-IFMA and AVX-512VL.
//! Runtime CPU feature detection is performed in
//! [`Module::new()`](poulpy_hal::api::ModuleNew::new); missing runtime features
//! cause a descriptive panic.
//!
//! # Compile-time requirements
//!
//! Backends are opt-in through Cargo features and matching target features:
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+avx512f" \
//!     cargo build --release --features enable-avx512f
//!
//! RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
//!     cargo build --release --features enable-ifma
//! ```
//!
//! If neither feature is enabled, this crate compiles as an empty shell so the
//! workspace remains portable on machines without AVX-512. Code that imports
//! AVX-512 backend types must enable the feature that exports them.
//!
//! # Correctness guarantees
//!
//! Operations are deterministic across runs. FFT operations are constrained to
//! preserve the rounding behavior expected by the reference backend, while NTT
//! operations are exact modulo their CRT prime sets.
//!
//! Integer overflow in limb arithmetic is intentional where the bivariate
//! representation relies on wrapping arithmetic to propagate carries correctly
//! across base-2^k limbs.
//!
//! # Safety invariants
//!
//! Unsafe kernels require:
//!
//! - the selected backend's CPU features to be enabled and present at runtime,
//! - input and output layouts to have matching shapes and documented bounds,
//! - buffers to satisfy the alignment required by `poulpy_hal::DEFAULTALIGN`.
//!
//! Violating those invariants may cause undefined behavior, panics, or silent
//! arithmetic errors.
//!
//! # Threading and concurrency
//!
//! Backend marker types are zero-sized and `Send + Sync`. `Module<BE>` values
//! hold immutable precomputed tables after construction. Operations take
//! mutable output references, so normal Rust borrowing rules prevent data races
//! at the API boundary.
//!
//! # Feature flags
//!
//! - `enable-avx512f`: exports `FFT64Avx512` and `NTT4x30Avx512`.
//! - `enable-ifma`: implies `enable-avx512f` and also exports `NTT3x42Ifma`.
//! - `enable-rayon`: implies `enable-avx512f` and exports `FFT64Avx512Rayon` and `NTT4x30Avx512Rayon`;
//!   with `enable-ifma`, it also exports `NTT3x42IfmaRayon`.
//! - `enable-ckks`: wires these backends into `poulpy-ckks` defaults.
//!
//! # Platform support
//!
//! - Required: x86-64.
//! - `FFT64Avx512`: AVX-512F + AVX2 + FMA.
//! - `NTT4x30Avx512`: AVX-512F.
//! - `NTT3x42Ifma`: AVX-512F + AVX-512-IFMA + AVX-512VL.
//! - Non-x86 targets and x86-64 CPUs without the selected feature set are not supported.
//!
//! # Usage
//!
//! The public backend marker types are used as type parameters to HAL, core,
//! CKKS, and bin-FHE generic APIs. Application code usually selects one of
//! these types in the backend-owning crate or benchmark harness.
//!
//! # Versioning and stability
//!
//! The public API consists of the backend marker types, FFT table wrappers, and
//! the `ntt3x42_ifma_api` support exports used by benchmarks. Other items are
//! implementation details.

#[cfg(all(feature = "enable-avx512f", not(docsrs), not(target_arch = "x86_64")))]
compile_error!("feature `enable-avx512f` requires target_arch = \"x86_64\".");

#[cfg(all(
    feature = "enable-avx512f",
    not(docsrs),
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
compile_error!("feature `enable-avx512f` requires AVX512F. Build with RUSTFLAGS=\"-C target-feature=+avx512f\".");

#[cfg(all(
    feature = "enable-ifma",
    not(docsrs),
    target_arch = "x86_64",
    not(target_feature = "avx512ifma")
))]
compile_error!(
    "feature `enable-ifma` requires AVX512-IFMA. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl\"."
);

#[cfg(all(
    feature = "enable-ifma",
    not(docsrs),
    target_arch = "x86_64",
    not(target_feature = "avx512vl")
))]
compile_error!(
    "feature `enable-ifma` requires AVX512VL. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl\"."
);

#[cfg(feature = "enable-rayon")]
mod execution;
#[cfg(feature = "enable-avx512f")]
mod fft64;
#[cfg(feature = "enable-avx512f")]
#[cfg(feature = "enable-avx512f")]
mod hal_impl;
#[cfg(feature = "enable-avx512f")]
mod ntt4x30_avx512;
#[cfg(feature = "enable-avx512f")]
mod znx_avx512;

#[cfg(feature = "enable-avx512f")]
mod vec_znx_big_avx512;

#[cfg(feature = "enable-ifma")]
mod ntt3x42_ifma;

#[cfg(all(feature = "enable-avx512f", feature = "enable-rayon"))]
pub use fft64::FFT64Avx512Rayon;
#[cfg(feature = "enable-avx512f")]
pub use fft64::{FFT64Avx512, FFT64Avx512ReimTable, ReimFFTAvx512, ReimIFFTAvx512};
#[cfg(feature = "enable-ifma")]
pub use ntt3x42_ifma::NTT3x42Ifma;
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
pub use ntt3x42_ifma::NTT3x42IfmaRayon;
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
#[doc(hidden)]
pub use ntt3x42_ifma::NTT3x42IfmaRayonExecutor;
#[cfg(feature = "enable-avx512f")]
pub use ntt4x30_avx512::NTT4x30Avx512;
#[cfg(all(feature = "enable-avx512f", feature = "enable-rayon"))]
pub use ntt4x30_avx512::NTT4x30Avx512Rayon;

/// Public surface for tools that drive [`NTT3x42Ifma`] kernels directly (e.g. the
/// benches): the precomputed twiddle tables, the prime set, and the
/// [`Ntt3x42IfmaDFTExecute`](ntt3x42_ifma_api::Ntt3x42IfmaDFTExecute) trait used to
/// dispatch a forward / inverse NTT.
///
/// The scalar test oracles for the IFMA SIMD kernels live under
/// `crate::ntt3x42_ifma::reference` and are not re-exported.
#[cfg(feature = "enable-ifma")]
pub mod ntt3x42_ifma_api {
    pub use crate::ntt3x42_ifma::primes::{PrimeSetNtt3x42Ifma, Primes42};
    pub use crate::ntt3x42_ifma::tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv};
    pub use crate::ntt3x42_ifma::traits::Ntt3x42IfmaDFTExecute;
}

#[cfg(all(feature = "enable-ckks", not(any(feature = "enable-avx512f", feature = "enable-ifma"))))]
compile_error!(
    "feature `enable-ckks` requires `enable-avx512f` or `enable-ifma` (without them nothing is built and test runs are silently empty)."
);

#[cfg(all(feature = "enable-avx512f", feature = "enable-ckks"))]
mod ckks_impl;
#[cfg(feature = "enable-avx512f")]
mod core_impl;

#[cfg(all(test, feature = "enable-avx512f", feature = "enable-ckks"))]
mod tests;

#[cfg(feature = "enable-avx512f")]
mod layout_compat;
