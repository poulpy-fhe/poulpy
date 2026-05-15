//! NEON-accelerated CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides [`FFT64Neon`] and [`NTT120Neon`], backend implementations
//! for [`poulpy_hal`] that target AArch64 NEON (ASIMD).
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the
//! [`Backend`](poulpy_hal::layouts::Backend) trait and a family of
//! _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for [`FFT64Neon`] and [`NTT120Neon`] using
//! hand-tuned NEON intrinsics where profitable, and falls back to the
//! portable reference kernels from `poulpy-cpu-ref` for ops where the
//! compiler-generated code is already near-optimal.
//!
//! The internal modules are organized by operation domain, mirroring the
//! AVX backend layout:
//!
//! | Module        | Domain                                                    |
//! |---------------|-----------------------------------------------------------|
//! | `module`      | Backend handle lifecycle, FFT/NTT table management        |
//! | `znx_neon`    | Single ring element (`Z[X]/(X^n+1)`) SIMD arithmetic      |
//! | `fft64/reim`  | Real/imaginary interleaved FFT primitives                 |
//! | `fft64/reim4` | 4-block Reim arithmetic for SVP/VMP/convolution           |
//! | `ntt120`      | Q120 NTT forward/inverse, lazy-modular q120b arithmetic   |
//!
//! # Scalar types
//!
//! - `FFT64Neon`: `ScalarPrep = f64`, `ScalarBig = i64`.
//! - `NTT120Neon`: `ScalarPrep = Q120bScalar`, `ScalarBig = i128`.
//!
//! # Platform support
//!
//! - **Required**: AArch64 (Apple Silicon, ARMv8-A and later). NEON/ASIMD
//!   is part of the AArch64 baseline, so no runtime feature detection is
//!   needed.
//! - **Not supported**: 32-bit ARM, x86, RISC-V, or any other architecture.
//!
//! # Feature flags
//!
//! - `enable-neon` (required): opt-in compilation of the backend. Without
//!   this feature, the crate is an empty shell, allowing the workspace to
//!   build on any target.
//! - `enable-ckks` (optional): wires the CKKS scheme OEP impls.
//!
//! # Build
//!
//! ```text
//! cargo build -p poulpy-cpu-arm --features enable-neon
//! cargo test  -p poulpy-cpu-arm --features enable-neon
//! ```

// ─────────────────────────────────────────────────────────────
// Build the backend only when the user opts in.
// `enable-neon` itself is an aarch64-only feature.
// ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "enable-neon", not(target_arch = "aarch64")))]
compile_error!("feature `enable-neon` requires target_arch = \"aarch64\".");

#[cfg(all(feature = "enable-neon", feature = "enable-ckks"))]
mod ckks_impl;
#[cfg(feature = "enable-neon")]
mod core_impl;
#[cfg(feature = "enable-neon")]
mod fft64;
#[cfg(feature = "enable-neon")]
mod hal_impl;
#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
mod neon;
#[cfg(feature = "enable-neon")]
mod ntt120;
#[cfg(all(test, feature = "enable-neon", feature = "enable-ckks"))]
mod tests;

#[cfg(feature = "enable-neon")]
pub use fft64::{FFT64Neon, FFT64NeonReimTable, ReimFFTNeon, ReimIFFTNeon};
#[cfg(feature = "enable-neon")]
pub use ntt120::NTT120Neon;

// --- TransferFrom impls ---
#[cfg(feature = "enable-neon")]
mod transfer_impls {
    use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
    use poulpy_hal::layouts::{Backend, TransferFrom};

    use crate::{FFT64Neon, NTT120Neon};

    impl TransferFrom<FFT64Neon> for FFT64Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Neon::from_host_bytes(&FFT64Neon::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for FFT64Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Neon::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }

    impl TransferFrom<NTT120Neon> for NTT120Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Neon::from_host_bytes(&NTT120Neon::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Ref> for NTT120Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Neon::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }

    // Cross-family: coefficient-domain buffers are compatible.
    // Prepared layouts must not be transferred directly; transfer the
    // non-prepared form and re-prepare on the destination backend.
    impl TransferFrom<NTT120Ref> for FFT64Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Neon::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<NTT120Neon> for FFT64Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Neon::from_host_bytes(&NTT120Neon::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for NTT120Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Neon::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Neon> for NTT120Neon {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Neon::from_host_bytes(&FFT64Neon::to_host_bytes(src))
        }
    }
}
