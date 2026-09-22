//! CI backends share concrete operation bindings and select CI transform plans.

#![allow(
    clippy::duplicate_mod,
    reason = "Each ring binds the shared implementations to distinct backend types."
)]

use crate::ring::ConjugateInvariant as Ring;

#[path = "ci/fft64.rs"]
pub(crate) mod fft64;
pub use fft64::FFT64Ref;
#[path = "ci/ntt4x30.rs"]
pub(crate) mod ntt4x30;
pub use ntt4x30::NTT4x30Ref;
#[cfg(feature = "enable-ckks")]
#[path = "ckks_impl.rs"]
mod ckks_impl;
#[cfg(feature = "enable-core")]
#[path = "core_impl.rs"]
mod core_impl;
#[path = "hal_impl/bindings.rs"]
mod hal_impl;
