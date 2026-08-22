//! NEON/ASIMD-accelerated CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides serial and Rayon-scheduled FFT64 and NTT4x30 backends
//! that leverage AArch64 SIMD (NEON / ASIMD)
//! to accelerate cryptographic operations in fully homomorphic encryption (FHE)
//! schemes based on Module-LWE.
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for `FFT64Neon` and `NTT4x30Neon` using hand-tuned NEON intrinsics
//! and inline assembly where profiling demonstrates performance benefits over compiler-generated code.
//!
//! The internal modules are organized by operation domain:
//!
//! | Module          | Domain                                                    |
//! |-----------------|-----------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, FFT/NTT table management        |
//! | `neon`          | Low-level NEON kernels (FFT, NTT, mat-vec, conversions)   |
//! | `fft64`         | f64 complex-FFT backend wiring (`FFT64Neon`)              |
//! | `ntt4x30`        | Q120 NTT backend wiring (`NTT4x30Neon`), VMP                |
//! | `znx`           | Single ring element (`Z[X]/(X^n+1)`) SIMD arithmetic      |
//! | `vec_znx_big`   | Large-coefficient (i128) ring element vectors             |
//!
//! # Scalar types
//!
//! - `FFT64Neon`: `DftWord = f64`, `BigWord = i64`.
//! - `NTT4x30Neon`: `DftWord = Q120bScalar` (4 × u64 CRT residues over `Primes30`), `BigWord = i128`.
//!
//! # CPU requirements
//!
//! This backend **requires** AArch64 CPUs. NEON / ASIMD is part of the AArch64
//! architectural baseline, so no runtime feature detection is needed: any
//! AArch64 target supports the full SIMD instruction set used by this crate.
//!
//! # Compile-time requirements
//!
//! NEON / ASIMD is enabled by default on AArch64 targets, so no `RUSTFLAGS`
//! target-feature flag is required:
//!
//! ```text
//! cargo build --features enable-neon
//! ```
//!
//! If `enable-neon` is enabled but the target architecture is not `aarch64`,
//! the build fails immediately with a `compile_error!`.
//!
//! # Correctness guarantees
//!
//! ## Determinism
//!
//! Integer / modular operations (`Znx*`, `I128BigOps`, `Ntt*`, `NttDFTExecute`) produce
//! **bit-identical results** against `poulpy-cpu-ref`. Floating-point operations in FFT
//! match the reference within ULP tolerance — NEON kernels use FMA (`vfmaq_f64`) where
//! the scalar reference does not, so individual rounding bits may differ.
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
//! `poulpy_hal::DEFAULTALIGN`. This alignment enables aligned SIMD loads/stores and
//! the use of `stnp` non-temporal pair stores in the VMP overwrite-apply path.
//!
//! ## Safety invariants
//!
//! Many functions are marked `unsafe` and require:
//! - Target architecture is AArch64 (verified at compile time via `compile_error!`).
//! - Input slices have matching lengths where documented.
//! - Input values satisfy documented bounds (e.g., `|x| < 2^50` for IEEE 754 conversions).
//! - Buffers are properly aligned (enforced by HAL allocators).
//!
//! Violating these invariants may result in:
//! - Undefined behavior (e.g., out-of-bounds memory access).
//! - Silent incorrect results (e.g., exceeding numeric bounds in FP conversion).
//! - Panics (in debug mode via assertions, or unconditionally for critical invariants).
//!
//! # Performance characteristics
//!
//! ## Asymptotic complexity
//!
//! - **FFT/IFFT**: O(n log n) for polynomial degree n.
//! - **Convolution**: O(n log n) via FFT-based approach.
//! - **VMP**: O(n · nrows · ncols) over a prime-major prepared-matrix layout.
//! - **Normalization**: O(n) per limb with vectorized digit extraction.
//!
//! ## Speedup over reference backend
//!
//! Speedups depend on the host micro-architecture and on the operation profile of the
//! workload. Run the benches in `poulpy-bench` (or the bundled `bench_neon_vs_ref` example)
//! on the target host for representative numbers. Qualitative trends:
//!
//! - **Ring element arithmetic** (add/sub/negate): bandwidth-bound, modest gains.
//! - **NTT/INTT, mat-vec, and convolution**: noticeable gains from hand-tuned NEON kernels;
//!   the convolution apply path dispatches to the fused canonical `ntt_mul_bbc_tile4_x2` kernel.
//! - **VMP** (large degree): the largest gains, scaling with coefficient size;
//!   uses `stnp` non-temporal stores to avoid cache pollution.
//!
//! ## Memory layout
//!
//! - **Vectorized storage**: Elements packed in groups of 4 (matching the AVX backend's
//!   `i64` block stride for cross-backend layout compatibility). NEON's 128-bit registers
//!   process each block as two `int64x2_t` / `uint64x2_t` per iteration.
//! - **Tail handling**: Scalar fallback for lengths not divisible by 4.
//! - **Cache-friendly**: 64-byte alignment ensures single cache line per vector load.
//! - **Q120 layout**: a q120 vector packs four u64 lanes (one per `Primes30` prime) across
//!   two NEON registers — `lo` for primes 0/1, `hi` for primes 2/3.
//!
//! # Threading and concurrency
//!
//! - **All backend markers are `Send + Sync`**: zero-sized marker types, no internal state.
//! - **`Module<FFT64Neon>` / `Module<NTT4x30Neon>` are `Send + Sync`**: FFT/NTT tables are immutable after construction.
//! - **Operations require `&mut` for outputs**: prevents data races at the API level.
//! - **No internal locking**: all synchronization is the caller's responsibility.
//!
//! # Feature flags
//!
//! - `enable-neon` (required): opt-in compilation of the backend. Without this feature,
//!   the crate is an empty shell, allowing the workspace to build on non-aarch64 targets.
//! - `enable-rayon` (optional): exports `FFT64NeonRayon` and `NTT4x30NeonRayon`.
//! - `enable-ckks` (optional): wires the CKKS scheme OEP impls for enabled backends.
//!
//! # Platform support
//!
//! - **Required**: AArch64 (Apple Silicon, ARMv8-A and later).
//! - **Tested under**: native AArch64 hosts.
//! - **Not supported**: 32-bit ARM, x86, RISC-V, or any other architecture.
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
//! This crate exports serial and Rayon marker types used as type
//! parameters to the HAL generic types. Application code typically does not import this
//! crate directly, but instead uses it via `poulpy_core` or `poulpy_bin_fhe` with
//! runtime backend selection.
//!
//! # Versioning and stability
//!
//! This crate follows semantic versioning. The public API consists of the backend marker
//! types, the `FFT64NeonReimTable` and `ReimFFT(I)Neon` FFT executors,
//! and their trait implementations from `poulpy_hal::oep`. All other items are
//! implementation details subject to change without notice.

#[cfg(all(feature = "enable-neon", not(target_arch = "aarch64")))]
compile_error!("feature `enable-neon` requires target_arch = \"aarch64\".");

#[cfg(all(feature = "enable-neon", feature = "enable-ckks"))]
mod ckks_impl;
#[cfg(feature = "enable-neon")]
mod core_impl;
#[cfg(feature = "enable-rayon")]
mod execution;
#[cfg(feature = "enable-neon")]
mod fft64;
#[cfg(feature = "enable-neon")]
mod hal_impl;
#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
mod neon;
#[cfg(feature = "enable-neon")]
mod ntt4x30;
#[cfg(all(test, feature = "enable-neon"))]
mod tests;

#[cfg(feature = "enable-rayon")]
pub use fft64::FFT64NeonRayon;
#[cfg(feature = "enable-neon")]
pub use fft64::{FFT64Neon, FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};
#[cfg(feature = "enable-neon")]
pub use ntt4x30::NTT4x30Neon;
#[cfg(feature = "enable-rayon")]
pub use ntt4x30::NTT4x30NeonRayon;

#[cfg(feature = "enable-neon")]
mod layout_compat;
