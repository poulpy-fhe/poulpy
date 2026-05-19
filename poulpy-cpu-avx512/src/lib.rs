//! AVX-512 / AVX-512-IFMA accelerated CPU backends for the Poulpy lattice cryptography library.
//!
//! This crate provides three backend implementations for [`poulpy_hal`]:
//!
//! - `FFT64Avx512`: f64 FFT backend, gated on `enable-avx512f`.
//! - `NTT120Avx512`: Q120 NTT backend over four ~30-bit CRT primes, gated on `enable-avx512f`.
//! - `NTT126Ifma`: Q126 NTT backend over three ~42-bit CRT primes, gated on `enable-ifma`.
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
//! | `ntt120_avx512`    | `NTT120Avx512` NTT, VMP, convolution, and DFT kernels      |
//! | `ntt126_ifma`      | `NTT126Ifma` IFMA NTT, VMP, SVP, convolution, and DFT code |
//! | `hal_impl`         | HAL OEP implementations and default wiring                 |
//! | `vec_znx_big_avx512` | AVX-512F i128 accumulator helpers                        |
//!
//! # Scalar types
//!
//! - `FFT64Avx512`: `ScalarPrep = f64`, `ScalarBig = i64`.
//! - `NTT120Avx512`: `ScalarPrep = Q120bScalar`, `ScalarBig = i128`.
//! - `NTT126Ifma`: `ScalarPrep = Q120bScalar`, `ScalarBig = i128`.
//!
//! # CPU requirements
//!
//! `FFT64Avx512` and `NTT120Avx512` require x86-64 with AVX-512F. The FFT64
//! backend also uses AVX2 and FMA kernels and checks those features at module
//! construction.
//!
//! `NTT126Ifma` additionally requires AVX-512-IFMA, AVX-512VL, BMI2, and ADX.
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
//! RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
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
//! - `enable-avx512f`: exports `FFT64Avx512` and `NTT120Avx512`.
//! - `enable-ifma`: implies `enable-avx512f` and also exports `NTT126Ifma`.
//! - `enable-ckks`: wires these backends into `poulpy-ckks` defaults.
//!
//! # Platform support
//!
//! - Required: x86-64.
//! - `FFT64Avx512`: AVX-512F + AVX2 + FMA.
//! - `NTT120Avx512`: AVX-512F.
//! - `NTT126Ifma`: AVX-512F + AVX-512-IFMA + AVX-512VL + BMI2 + ADX.
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
//! the `ntt126_ifma_api` support exports used by benchmarks. Other items are
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
    "feature `enable-ifma` requires AVX512-IFMA. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx\"."
);

#[cfg(all(
    feature = "enable-ifma",
    not(docsrs),
    target_arch = "x86_64",
    not(target_feature = "avx512vl")
))]
compile_error!(
    "feature `enable-ifma` requires AVX512VL. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx\"."
);

#[cfg(all(feature = "enable-ifma", not(docsrs), target_arch = "x86_64", not(target_feature = "bmi2")))]
compile_error!(
    "feature `enable-ifma` requires BMI2. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx\"."
);

#[cfg(all(feature = "enable-ifma", not(docsrs), target_arch = "x86_64", not(target_feature = "adx")))]
compile_error!(
    "feature `enable-ifma` requires ADX. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx\"."
);

#[cfg(feature = "enable-avx512f")]
mod fft64;
#[cfg(feature = "enable-avx512f")]
mod hal_impl;
#[cfg(feature = "enable-avx512f")]
mod ntt120_avx512;
#[cfg(feature = "enable-avx512f")]
mod znx_avx512;

#[cfg(feature = "enable-avx512f")]
mod vec_znx_big_avx512;

#[cfg(feature = "enable-ifma")]
mod ntt126_ifma;

#[cfg(feature = "enable-avx512f")]
pub use fft64::{FFT64Avx512, FFT64Avx512ReimTable, ReimFFTAvx512, ReimIFFTAvx512};
#[cfg(feature = "enable-avx512f")]
pub use ntt120_avx512::NTT120Avx512;
#[cfg(feature = "enable-ifma")]
pub use ntt126_ifma::NTT126Ifma;

/// Public surface for tools that drive [`NTT126Ifma`] kernels directly (e.g. the
/// benches): the precomputed twiddle tables, the prime set, and the
/// [`Ntt126IfmaDFTExecute`](ntt126_ifma_api::Ntt126IfmaDFTExecute) trait used to
/// dispatch a forward / inverse NTT.
///
/// The scalar test oracles for the IFMA SIMD kernels live under
/// `crate::ntt126_ifma::reference` and are not re-exported.
#[cfg(feature = "enable-ifma")]
pub mod ntt126_ifma_api {
    pub use crate::ntt126_ifma::primes::{PrimeSetNtt126Ifma, Primes42};
    pub use crate::ntt126_ifma::tables::{Ntt126IfmaTable, Ntt126IfmaTableInv};
    pub use crate::ntt126_ifma::traits::Ntt126IfmaDFTExecute;
}

#[cfg(all(feature = "enable-avx512f", feature = "enable-ckks"))]
mod ckks_impl;
#[cfg(feature = "enable-avx512f")]
mod core_impl;

#[cfg(all(test, feature = "enable-avx512f", feature = "enable-ckks"))]
mod tests;

// --- TransferFrom impls ---
#[cfg(feature = "enable-avx512f")]
mod transfer_impls {
    use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
    use poulpy_hal::layouts::{Backend, TransferFrom};

    #[cfg(feature = "enable-ifma")]
    use crate::NTT126Ifma;
    use crate::{FFT64Avx512, NTT120Avx512};

    impl TransferFrom<FFT64Avx512> for FFT64Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx512::from_host_bytes(&FFT64Avx512::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for FFT64Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx512::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }

    impl TransferFrom<NTT120Avx512> for NTT120Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx512::from_host_bytes(&NTT120Avx512::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Ref> for NTT120Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx512::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }

    // Cross-family: coefficient-domain buffers are compatible.
    // Prepared layouts must not be transferred directly; transfer the
    // non-prepared form and re-prepare on the destination backend.
    impl TransferFrom<NTT120Ref> for FFT64Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx512::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Avx512> for FFT64Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx512::from_host_bytes(&NTT120Avx512::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for NTT120Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx512::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Avx512> for NTT120Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx512::from_host_bytes(&FFT64Avx512::to_host_bytes(src))
        }
    }

    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<NTT126Ifma> for NTT126Ifma {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT126Ifma::from_host_bytes(&NTT126Ifma::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<NTT120Ref> for NTT126Ifma {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT126Ifma::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<FFT64Ref> for NTT126Ifma {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT126Ifma::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<NTT120Avx512> for NTT126Ifma {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT126Ifma::from_host_bytes(&NTT120Avx512::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<FFT64Avx512> for NTT126Ifma {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT126Ifma::from_host_bytes(&FFT64Avx512::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<NTT126Ifma> for FFT64Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx512::from_host_bytes(&NTT126Ifma::to_host_bytes(src))
        }
    }
    #[cfg(feature = "enable-ifma")]
    impl TransferFrom<NTT126Ifma> for NTT120Avx512 {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx512::from_host_bytes(&NTT126Ifma::to_host_bytes(src))
        }
    }
}
