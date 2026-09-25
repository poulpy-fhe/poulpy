//! CI backends share concrete operation bindings and select CI transform plans.

#![allow(
    clippy::duplicate_mod,
    reason = "Each ring binds the shared implementations to distinct backend types."
)]

use poulpy_cpu_ref::ring::ConjugateInvariant as Ring;

#[path = "ci/fft64.rs"]
pub(crate) mod fft64;
pub use fft64::FFT64Neon;
#[cfg(feature = "enable-rayon")]
pub use fft64::FFT64NeonRayon;
#[path = "ci/ntt4x30.rs"]
pub(crate) mod ntt4x30;
pub use ntt4x30::NTT4x30Neon;
#[cfg(feature = "enable-rayon")]
pub use ntt4x30::NTT4x30NeonRayon;
#[path = "core_impl.rs"]
mod core_impl;
#[path = "hal_impl.rs"]
mod hal_impl;
