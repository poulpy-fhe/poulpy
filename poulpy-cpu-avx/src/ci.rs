//! CI backends share concrete operation bindings and select CI transform plans.

#![allow(
    clippy::duplicate_mod,
    reason = "Each ring binds the shared implementations to distinct backend types."
)]

use poulpy_cpu_ref::ring::ConjugateInvariant as Ring;

#[path = "ci/fft64.rs"]
pub(crate) mod fft64;
pub use fft64::FFT64Avx;
#[cfg(feature = "enable-rayon")]
pub use fft64::FFT64AvxRayon;
#[path = "ci/ntt4x30.rs"]
pub(crate) mod ntt4x30;
pub use ntt4x30::NTT4x30Avx;
#[cfg(feature = "enable-rayon")]
pub use ntt4x30::NTT4x30AvxRayon;
#[path = "core_impl.rs"]
mod core_impl;
#[path = "hal_impl.rs"]
mod hal_impl;
