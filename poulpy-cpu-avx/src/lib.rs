//! AVX2/FMA-accelerated CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides `FFT64Avx`, a high-performance backend implementation for [`poulpy_hal`]
//! that leverages x86-64 SIMD instruction sets (AVX2 and FMA) to accelerate cryptographic operations
//! in fully homomorphic encryption (FHE) schemes based on Module-LWE.
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for the `FFT64Avx` backend using hand-optimized AVX2/FMA intrinsics
//! and assembly kernels where profiling demonstrates performance benefits over compiler-generated code.
//!
//! The internal modules are organized by operation domain:
//!
//! | Module          | Domain                                                    |
//! |-----------------|-----------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, FFT table management            |
//! | `scratch`       | Temporary memory allocation and arena-style sub-allocation|
//! | `znx_avx`       | Single ring element (`Z[X]/(X^n+1)`) SIMD arithmetic      |
//! | `vec_znx`       | Vectors of ring elements (limb decomposition)             |
//! | `vec_znx_big`   | Large-coefficient (multi-word) ring element vectors       |
//! | `vec_znx_dft`   | Fourier-domain ring element vectors (forward/inverse DFT) |
//! | `reim`          | Real/imaginary interleaved FFT primitives                 |
//! | `convolution`   | Polynomial convolution via FFT, by-constant, and pairwise |
//! | `svp`           | Scalar-vector product in frequency domain                 |
//!
//! # Scalar types
//!
//! For the `FFT64Avx` backend:
//!
//! - `ScalarPrep = f64`: coefficients in the DFT / frequency domain.
//! - `ScalarBig  = i64`: coefficients in the large-integer (multi-word) domain.
//!   meaning each coefficient occupies exactly one scalar word.
//!
//! # CPU requirements
//!
//! This backend **requires** x86-64 CPUs with:
//! - **AVX2**: 256-bit SIMD registers and operations
//! - **FMA**: Fused multiply-add for reduced rounding error in FFT
//!
//! Runtime CPU feature detection is performed in [`Module::new()`](poulpy_hal::api::ModuleNew::new).
//! If the required features are not present, the constructor panics with a descriptive error message.
//!
//! # Compile-time requirements
//!
//! To compile this crate, you must enable AVX2 and FMA target features:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --release
//! ```
//!
//! Or explicitly:
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
//! ```
//!
//! Failure to enable these features at compile time will result in a compilation error.
//!
//! # Correctness guarantees
//!
//! ## Determinism
//!
//! All operations produce **bit-identical results** across different runs and different backends
//! (when compared to `poulpy-cpu-ref`). Floating-point operations in FFT are constrained to
//! maintain error < 0.5 ULP, ensuring correct rounding when converting back to integers.
//!
//! ## Overflow handling
//!
//! Integer overflow is **intentional** and managed through bivariate polynomial representation.
//! The normalization functions (`znx_normalize_*`) use wrapping arithmetic to propagate carries
//! correctly across limbs in base-2^k representation.
//!
//! ## Memory alignment
//!
//! All data layouts enforce 64-byte alignment (matching cache line size) as specified by
//! `poulpy_hal::DEFAULTALIGN`. This alignment is verified at buffer allocation and enables
//! the use of aligned SIMD loads/stores for maximum performance.
//!
//! ## Safety invariants
//!
//! Many functions are marked `unsafe` and require:
//! - CPU features (AVX2/FMA) are present (verified at module creation)
//! - Input slices have matching lengths where documented
//! - Input values satisfy documented bounds (e.g., `|x| < 2^50` for IEEE 754 conversions)
//! - Buffers are properly aligned (enforced by HAL allocators)
//!
//! Violating these invariants may result in:
//! - Undefined behavior (e.g., invalid SIMD instructions, out-of-bounds memory access)
//! - Silent incorrect results (e.g., exceeding numeric bounds in FP conversion)
//! - Panics (in debug mode via assertions, or unconditionally for critical invariants)
//!
//! # Performance characteristics
//!
//! ## Asymptotic complexity
//!
//! - **FFT/IFFT**: O(n log n) for polynomial degree n
//! - **Convolution**: O(n log n) via FFT-based approach
//! - **Normalization**: O(n) per limb with vectorized digit extraction
//!
//! ## Speedup over reference backend
//!
//! Speedups depend on the host micro-architecture and on the operation profile of the
//! workload. Run the benches in `poulpy-bench` on the target host for representative
//! numbers. Qualitative trends:
//!
//! - **Ring element arithmetic** (add/sub/negate): bandwidth-bound, modest gains.
//! - **FFT16 kernels** (hand-written assembly): noticeably ahead of compiler-generated
//!   intrinsics on supported micro-architectures.
//! - **Convolution** (large degree): the largest gains, scaling with coefficient size.
//!
//! ## Memory layout
//!
//! - **Vectorized storage**: Elements packed in groups of 4 (matching AVX2 register width for `i64`)
//! - **Tail handling**: Scalar fallback for lengths not divisible by 4
//! - **Cache-friendly**: 64-byte alignment ensures single cache line per vector load
//!
//! # Threading and concurrency
//!
//! - **`FFT64Avx` is `Send + Sync`**: Zero-sized marker type, no internal state.
//! - **`Module<FFT64Avx>` is `Send + Sync`**: FFT tables are immutable after construction.
//! - **Operations require `&mut` for outputs**: Prevents data races at the API level.
//! - **No internal locking**: All synchronization is caller's responsibility.
//!
//! # Feature flags
//!
//! - `enable-avx` (optional): Historically used for conditional compilation, currently inactive.
//!
//! # Platform support
//!
//! - **Required**: x86-64 architecture with AVX2 and FMA
//! - **Tested on**: Linux (x86_64), macOS (Intel), Windows (x86_64)
//! - **Not supported**: ARM, RISC-V, or other architectures
//!
//! # Threat model
//!
//! This library assumes an **"honest but curious"** adversary model:
//! - **No malicious inputs**: Callers are trusted to provide well-formed data within documented bounds.
//! - **No timing attack mitigation**: Operations are not constant-time (performance is prioritized).
//! - **Memory safety**: Bounds are validated to prevent crashes and corruption, but not for security.
//!
//! # Usage
//!
//! This crate exports a single public type, `FFT64Avx`, which is used as a type parameter
//! to the HAL generic types. Application code typically does not import this crate directly,
//! but instead uses it via `poulpy_core` or `poulpy_bin_fhe` with runtime backend selection.
//!
//! # Versioning and stability
//!
//! This crate follows semantic versioning. The public API consists solely of the `FFT64Avx`
//! marker type and its trait implementations from `poulpy_hal::oep`. All other items are
//! implementation details subject to change without notice.

// ─────────────────────────────────────────────────────────────
// Build the backend **only when ALL conditions are satisfied**
// ─────────────────────────────────────────────────────────────
//#![cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]

// If the user enables this backend but targets a non-x86_64 CPU → abort
#[cfg(all(feature = "enable-avx", not(target_arch = "x86_64")))]
compile_error!("feature `enable-avx` requires target_arch = \"x86_64\".");

// If the user enables this backend but AVX2 isn't enabled in the target → abort
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", not(target_feature = "avx2")))]
compile_error!("feature `enable-avx` requires AVX2. Build with RUSTFLAGS=\"-C target-feature=+avx2\".");

// If the user enables this backend but FMA isn't enabled in the target → abort
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", not(target_feature = "fma")))]
compile_error!("feature `enable-avx` requires FMA. Build with RUSTFLAGS=\"-C target-feature=+fma\".");

// Keep the crate as a true opt-in backend: without `enable-avx`, none of the
// AVX modules or their unit tests are compiled.
#[cfg(all(feature = "enable-avx", feature = "enable-ckks"))]
mod ckks_impl;
#[cfg(feature = "enable-avx")]
mod core_impl;
#[cfg(feature = "enable-avx")]
mod fft64;
#[cfg(feature = "enable-avx")]
mod hal_impl;
#[cfg(feature = "enable-avx")]
mod ntt120;
#[cfg(all(test, feature = "enable-avx", feature = "enable-ckks"))]
mod tests;
#[cfg(feature = "enable-avx")]
mod znx_avx;

#[cfg(feature = "enable-avx")]
pub use fft64::{FFT64Avx, FFT64AvxReimTable, ReimFFTAvx, ReimIFFTAvx};
#[cfg(feature = "enable-avx")]
pub use ntt120::NTT120Avx;

// --- TransferFrom impls ---
#[cfg(feature = "enable-avx")]
mod transfer_impls {
    use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
    use poulpy_hal::layouts::{Backend, TransferFrom};

    use crate::{FFT64Avx, NTT120Avx};

    impl TransferFrom<FFT64Avx> for FFT64Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx::from_host_bytes(&FFT64Avx::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for FFT64Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }

    impl TransferFrom<NTT120Avx> for NTT120Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx::from_host_bytes(&NTT120Avx::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Ref> for NTT120Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }

    // Cross-family: coefficient-domain buffers are compatible.
    // Prepared layouts must not be transferred directly; transfer the
    // non-prepared form and re-prepare on the destination backend.
    impl TransferFrom<NTT120Ref> for FFT64Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Avx> for FFT64Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Avx::from_host_bytes(&NTT120Avx::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for NTT120Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Avx> for NTT120Avx {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Avx::from_host_bytes(&FFT64Avx::to_host_bytes(src))
        }
    }
}
