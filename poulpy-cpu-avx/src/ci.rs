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
#[cfg(feature = "enable-ckks")]
#[path = "ckks_impl.rs"]
mod ckks_impl;
#[cfg(feature = "enable-ckks")]
#[path = "ckks_mod_up.rs"]
mod ckks_mod_up;
#[path = "core_impl.rs"]
mod core_impl;
#[path = "hal_impl.rs"]
mod hal_impl;

#[cfg(feature = "enable-ckks")]
use crate::FFT64AvxReimTable;
