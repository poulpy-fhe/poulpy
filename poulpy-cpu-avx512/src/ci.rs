//! CI backends share concrete operation bindings and select CI transform plans.

#![allow(
    clippy::duplicate_mod,
    reason = "Each ring binds the shared implementations to distinct backend types."
)]

use poulpy_cpu_ref::ring::ConjugateInvariant as Ring;

#[path = "ci/fft64.rs"]
pub(crate) mod fft64;
pub use fft64::FFT64Avx512;
#[cfg(feature = "enable-rayon")]
pub use fft64::FFT64Avx512Rayon;
#[path = "ci/ntt4x30_avx512.rs"]
pub(crate) mod ntt4x30_avx512;
pub use ntt4x30_avx512::NTT4x30Avx512;
#[cfg(feature = "enable-rayon")]
pub use ntt4x30_avx512::NTT4x30Avx512Rayon;
#[cfg(feature = "enable-ifma")]
#[path = "ci/ntt3x42_ifma.rs"]
pub(crate) mod ntt3x42_ifma;
#[cfg(feature = "enable-ifma")]
pub use ntt3x42_ifma::NTT3x42Ifma;
#[cfg(feature = "enable-ifma")]
#[cfg(feature = "enable-rayon")]
pub use ntt3x42_ifma::NTT3x42IfmaRayon;
#[path = "core_impl.rs"]
mod core_impl;
#[path = "hal_impl.rs"]
mod hal_impl;
