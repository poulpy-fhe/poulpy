//! NEON-accelerated CPU backend for the Poulpy lattice cryptography library.
//!
//! Provides [`FFT64Neon`] and [`NTT120Neon`], gated on the `enable-neon` feature
//! (aarch64-only). With `enable-neon` disabled the crate is an empty shell.

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
