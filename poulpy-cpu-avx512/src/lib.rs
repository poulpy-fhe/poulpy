//! AVX-512 / AVX-512-IFMA accelerated CPU backends for the Poulpy lattice cryptography library.
//!
//! This crate provides three high-performance backend implementations for [`poulpy_hal`] that
//! leverage x86-64 SIMD instruction sets (AVX-512F and AVX-512-IFMA) to accelerate cryptographic
//! operations in fully homomorphic encryption (FHE) schemes based on Ring-Learning-With-Errors (RLWE):
//!
//! - `FFT64Avx512`: FFT64-domain backend using AVX-512F for REIM FFT kernels.
//! - `NTT120Avx512`: NTT-domain backend over four ~30-bit CRT primes, AVX-512F-only (no IFMA).
//! - `NTT126Ifma`: NTT-domain backend over three ~42-bit CRT primes, accelerated with AVX-512-IFMA.
//!
//! **Backend selection.** On hosts that support AVX-512-IFMA, prefer `NTT126Ifma`: its
//! `vpmadd52`-driven mat_vec / VMP / SVP kernels typically lead end-to-end on CKKS-style
//! workloads. `NTT120Avx512` is the right choice on AVX-512F hosts that lack IFMA
//! (e.g. Skylake-X / Cascade Lake / KNL).
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for `FFT64Avx512`, `NTT120Avx512`, and `NTT126Ifma` using
//! hand-optimized AVX-512 / IFMA intrinsics where profiling demonstrates performance
//! benefits over compiler-generated code, and reuses the reference backend for colder paths.
//!
//! The internal modules are organized by operation domain:
//!
//! | Module          | Domain                                                    |
//! |-----------------|-----------------------------------------------------------|
//! | `fft64`         | `FFT64Avx512` backend and AVX-512 REIM FFT kernels        |
//! | `znx_avx512`    | AVX-512F single ring element (`Z[X]/(X^n+1)`) arithmetic  |
//! | `ntt120_avx512` | `NTT120Avx512` backend (AVX-512F-only, 4×30-bit CRT primes) |
//! | `ntt126_ifma`   | `NTT126Ifma` backend, IFMA mat_vec, vec_znx_dft / vec_znx_big |
//! | `hal_impl`      | Implementations of the HAL OEP traits per backend         |
//!
//! # Scalar types
//!
//! - For `FFT64Avx512`:  `ScalarPrep = f64` (DFT-domain), `ScalarBig = i64` (one scalar word per limb).
//! - For `NTT120Avx512`: `ScalarPrep = Q120bScalar` (shared 4-lane prep scalar with four active ~30-bit CRT residues), `ScalarBig = i128`.
//! - For `NTT126Ifma`:   `ScalarPrep = Q120bScalar` (shared 4-lane prep scalar with three active ~42-bit CRT residues plus padding), `ScalarBig = i128`.
//!
//! # CPU requirements
//!
//! `FFT64Avx512` and `NTT120Avx512` require only:
//! - **AVX-512F**: 512-bit SIMD foundation.
//!
//! `NTT126Ifma` additionally requires:
//! - **AVX-512-IFMA**: 52-bit fused-multiply-add instructions (`vpmadd52`).
//! - **AVX-512VL**: 128/256-bit variable-length operations on the AVX-512 register file.
//!
//! Runtime CPU feature detection is performed in [`Module::new()`](poulpy_hal::api::ModuleNew::new).
//! If the required features are not present, the constructor panics with a descriptive error message.
//!
//! # Compile-time requirements
//!
//! Two layered cargo features control which backend is built:
//!
//! - `enable-avx512f` builds `FFT64Avx512` and `NTT120Avx512` (needs AVX-512F at compile time).
//! - `enable-ifma` (which implies `enable-avx512f`) additionally builds `NTT126Ifma`
//!   (needs AVX-512F + AVX-512-IFMA + AVX-512VL at compile time).
//!
//! ```text
//! # AVX-512F only host (Skylake-X / Cascade Lake / KNL): FFT64Avx512 + NTT120Avx512
//! RUSTFLAGS="-C target-feature=+avx512f" \
//!     cargo build --release --features enable-avx512f
//!
//! # IFMA-capable host (Ice Lake / Tiger Lake / Sapphire Rapids): both backends
//! RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
//!     cargo build --release --features enable-ifma
//!
//! # On a supported host:
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --features enable-ifma
//! ```
//!
//! Failure to enable the matching target features will result in a compile error.
//!
//! # Correctness guarantees
//!
//! ## Determinism
//!
//! All operations produce **bit-identical results** across different runs and different backends
//! (when compared to `poulpy-cpu-ref` and `poulpy-cpu-avx`). Floating-point operations in FFT are
//! constrained to maintain error < 0.5 ULP, ensuring correct rounding when converting back to
//! integers. NTT operations are exact modulo each CRT prime.
//!
//! ## Overflow handling
//!
//! Integer overflow is **intentional** and managed through bivariate polynomial representation.
//! The normalization functions (`znx_normalize_*`, `nfc_*`) use wrapping arithmetic to propagate
//! carries correctly across limbs in base-2^k representation.
//!
//! ## Memory alignment
//!
//! All data layouts enforce 64-byte alignment (matching cache line size) as specified by
//! `poulpy_hal::DEFAULTALIGN`. This alignment is verified at buffer allocation and enables
//! the use of aligned SIMD loads/stores for maximum performance on AVX-512.
//!
//! ## Safety invariants
//!
//! Many functions are marked `unsafe` and require:
//! - CPU features required by the selected backend are present
//!   (`FFT64Avx512` / `NTT120Avx512`: AVX-512F; `NTT126Ifma`: AVX-512F + AVX-512-IFMA + AVX-512VL),
//!   verified at module creation
//! - Input slices have matching lengths where documented
//! - Input values satisfy documented bounds (e.g., `|x| < 2^50` for IEEE 754 conversions,
//!   limb residues `< 2^52` for IFMA `vpmadd52` accumulators)
//! - Buffers are properly aligned (enforced by HAL allocators)
//!
//! Violating these invariants may result in:
//! - Undefined behavior (e.g., invalid SIMD instructions, out-of-bounds memory access)
//! - Silent incorrect results (e.g., exceeding numeric bounds in FP / IFMA conversion)
//! - Panics (in debug mode via assertions, or unconditionally for critical invariants)
//!
//! # Performance characteristics
//!
//! ## Asymptotic complexity
//!
//! - **FFT/IFFT** (`FFT64Avx512`): O(n log n) for polynomial degree n.
//! - **NTT/INTT** (`NTT120Avx512`, `NTT126Ifma`): O(n log n) for polynomial degree n.
//! - **Convolution**: O(n log n) via FFT- or NTT-based approach.
//! - **Normalization**: O(n) per limb with vectorized digit extraction.
//!
//! ## Speedup over reference / AVX2 backends
//!
//! Speedups depend strongly on the host micro-architecture (Intel vs AMD, generation,
//! cache hierarchy, AVX-512 implementation width) and on the operation profile of the
//! workload, so concrete factors are not quoted here. Run the benches in `poulpy-bench`
//! on the target host for representative numbers.
//!
//! Qualitative trends observed across operations:
//!
//! - **Ring element arithmetic** (add/sub/negate, lazy 4-lane prep-scalar ops, VecZnxBig i128 ops):
//!   memory-bandwidth bound; the 256→512-bit widening yields little headroom over AVX2.
//! - **NTT forward / inverse** (`NTT120Avx512`, 2-coefficient pair-pack): a moderate
//!   improvement over AVX2 in cache-resident regimes; the gap narrows at large `n` as
//!   the kernel becomes bandwidth-bound.
//! - **NTT and BBC mat_vec / VMP / SVP** (`NTT126Ifma`): IFMA's `vpmadd52` chain shortens
//!   the modular-multiply critical path and is the main beneficiary of AVX-512 on
//!   compute-heavy paths (key-switch, external product, relinearization).
//! - **FFT16 kernels** (hand-written assembly): on par with the AVX2 backend.
//!
//! Net on IFMA-capable hardware: `NTT126Ifma` is typically the faster choice end-to-end
//! on CKKS-style workloads, with the largest gains on VMP-dominated operations.
//!
//! ## Memory layout
//!
//! - **Vectorized storage**: Elements packed in groups of 8 (matching AVX-512 register width for `i64` / `f64`).
//! - **Tail handling**: Scalar fallback for lengths not divisible by 8.
//! - **Cache-friendly**: 64-byte alignment ensures single cache line per AVX-512 register load.
//!
//! # Threading and concurrency
//!
//! - **`FFT64Avx512`, `NTT120Avx512`, and `NTT126Ifma` are `Send + Sync`**: zero-sized marker
//!   types, no internal state.
//! - **`Module<FFT64Avx512>` / `Module<NTT120Avx512>` / `Module<NTT126Ifma>` are `Send + Sync`**:
//!   FFT/NTT tables are immutable after construction.
//! - **Operations require `&mut` for outputs**: prevents data races at the API level.
//! - **No internal locking**: all synchronization is the caller's responsibility.
//!
//! # Feature flags
//!
//! - `enable-avx512f`: builds `FFT64Avx512` and `NTT120Avx512`. Needs AVX-512F at compile time.
//! - `enable-ifma`: implies `enable-avx512f` and additionally builds `NTT126Ifma`. Needs
//!   AVX-512F + AVX-512-IFMA + AVX-512VL at compile time.
//!
//! When neither feature is enabled, the crate compiles to an empty shell so that workspaces
//! targeting non-AVX-512 platforms (e.g. macOS ARM, older x86) remain portable.
//!
//! # Platform support
//!
//! - **Required**: x86-64 architecture.
//! - **For `FFT64Avx512` / `NTT120Avx512`**: AVX-512F.
//! - **For `NTT126Ifma`**: AVX-512F + AVX-512-IFMA + AVX-512VL.
//! - **Tested on**: Linux (x86_64) on AVX-512-capable Intel and AMD CPUs.
//! - **Not supported**: ARM, RISC-V, non-x86_64 targets, or x86_64 CPUs lacking the
//!   required AVX-512 features for the selected backend.
//!
//! # Threat model
//!
//! This library assumes an **"honest but curious"** adversary model:
//! - **No malicious inputs**: callers are trusted to provide well-formed data within documented bounds.
//! - **No timing attack mitigation**: operations are not constant-time (performance is prioritized).
//! - **Memory safety**: bounds are validated to prevent crashes and corruption, but not for security.
//!
//! # Usage
//!
//! This crate exports three public types — `FFT64Avx512`, `NTT120Avx512`, and `NTT126Ifma` —
//! used as type parameters to the HAL generic types. Application code typically does not
//! import this crate directly, but instead uses it via `poulpy_core`, `poulpy_ckks` or `poulpy_bin_fhe`
//! with runtime backend selection.
//!
//! # Versioning and stability
//!
//! This crate follows semantic versioning. The public API consists solely of the
//! `FFT64Avx512`, `NTT120Avx512`, and `NTT126Ifma` marker types and their trait
//! implementations from `poulpy_hal::oep`. All other items are implementation details
//! subject to change without notice.

// ─────────────────────────────────────────────────────────────
// Build the backends only when ALL prerequisites are satisfied
// ─────────────────────────────────────────────────────────────

// `enable-avx512f` (gates `FFT64Avx512`) requires x86-64 + AVX-512F.
#[cfg(all(feature = "enable-avx512f", not(docsrs), not(target_arch = "x86_64")))]
compile_error!("feature `enable-avx512f` requires target_arch = \"x86_64\".");

#[cfg(all(
    feature = "enable-avx512f",
    not(docsrs),
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
compile_error!("feature `enable-avx512f` requires AVX512F. Build with RUSTFLAGS=\"-C target-feature=+avx512f\".");

// `enable-ifma` (gates `NTT126Ifma`) additionally requires AVX-512-IFMA and AVX-512VL.
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

// `FFT64Avx512`, `NTT120Avx512`, and their supporting AVX-512F `znx_avx512` helpers are gated on `enable-avx512f`.
#[cfg(feature = "enable-avx512f")]
mod fft64;
#[cfg(feature = "enable-avx512f")]
mod hal_impl;
#[cfg(feature = "enable-avx512f")]
mod ntt120_avx512;
#[cfg(feature = "enable-avx512f")]
mod znx_avx512;

// AVX-512F i128 kernels shared by both NTT120 backends.
#[cfg(feature = "enable-avx512f")]
mod vec_znx_big_avx512;

// `NTT126Ifma` and its IFMA-specific kernels are gated on `enable-ifma`.
#[cfg(feature = "enable-ifma")]
mod ntt126_ifma;

#[cfg(feature = "enable-avx512f")]
pub use fft64::{FFT64Avx512, ReimFFTAvx512, ReimIFFTAvx512};
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

#[cfg(feature = "enable-avx512f")]
use poulpy_core::oep::CoreImpl;

#[cfg(feature = "enable-avx512f")]
unsafe impl CoreImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_core::impl_core_default_methods!(FFT64Avx512);
}

#[cfg(feature = "enable-avx512f")]
unsafe impl CoreImpl<NTT120Avx512> for NTT120Avx512 {
    poulpy_core::impl_core_default_methods!(NTT120Avx512);
}

#[cfg(feature = "enable-ifma")]
unsafe impl CoreImpl<NTT126Ifma> for NTT126Ifma {
    poulpy_core::impl_core_default_methods!(NTT126Ifma);
}
