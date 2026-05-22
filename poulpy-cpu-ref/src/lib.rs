#![allow(clippy::too_many_arguments)]

//! Reference (portable) CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two reference implementations for [`poulpy_hal`]:
//!
//! - [`FFT64Ref`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT120Ref`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt120`] module.
//!
//! Both are canonical reference implementations: portable across all CPU architectures,
//! prioritising correctness and debuggability over throughput.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-core")]
#[doc(hidden)]
pub mod core_impl;
pub mod fft64;
pub mod hal_defaults;
mod hal_impl;
pub mod ntt120;
pub mod reference;

#[cfg(test)]
mod tests;

pub use poulpy_hal::cast_mut;

pub mod api {
    pub use poulpy_hal::api::*;
}

pub mod layouts {
    pub use poulpy_hal::layouts::*;
}

pub mod source {
    pub use poulpy_hal::source::*;
}

pub use fft64::{FFT64Ref, FFT64ReimTable};
pub use ntt120::{NTT120Ref, NTT120RefHandle};

// --- TransferFrom impls ---
mod transfer_impls {
    use poulpy_hal::layouts::{Backend, TransferFrom};

    use crate::{FFT64Ref, NTT120Ref};

    impl TransferFrom<FFT64Ref> for FFT64Ref {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Ref::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }

    impl TransferFrom<NTT120Ref> for NTT120Ref {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Ref::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }

    // Cross-family: coefficient-domain buffers are compatible (plain i64 data).
    // Prepared layouts are backend-specific and must not be transferred directly;
    // transfer the non-prepared form and re-prepare on the destination backend.
    impl TransferFrom<NTT120Ref> for FFT64Ref {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            FFT64Ref::from_host_bytes(&NTT120Ref::to_host_bytes(src))
        }
    }
    impl TransferFrom<FFT64Ref> for NTT120Ref {
        fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
            NTT120Ref::from_host_bytes(&FFT64Ref::to_host_bytes(src))
        }
    }
}
