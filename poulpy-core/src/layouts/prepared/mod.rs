//! DFT-domain (prepared) ciphertext and key layouts.
//!
//! Prepared variants store polynomials in the frequency domain of the
//! backend's DFT/NTT transform, enabling O(N log N) polynomial
//! multiplication via [`poulpy_hal`]'s `VmpApplyDftToDft` and
//! `SvpApplyDft` operations.
//!
//! Every prepared type is parametrised by `B: Backend` in addition
//! to `D: Data`, making it tied to a specific backend instance.
//! Prepared layouts are created from their standard counterparts
//! via `prepare` / `prepare_*` methods.

mod gglwe;
mod gglwe_bound;
mod gglwe_to_ggsw_key;
mod ggsw;
mod glwe;
mod glwe_automorphism_key;
mod glwe_public_key;
mod glwe_secret;
mod glwe_secret_tensor;
mod glwe_switching_key;
mod glwe_tensor_key;
mod glwe_to_lwe_key;
mod linear_transformation;
mod lwe_switching_key;
mod lwe_to_glwe_key;

pub use gglwe::*;
pub use gglwe_bound::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_public_key::*;
pub use glwe_secret::*;
pub use glwe_secret_tensor::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
pub use glwe_to_lwe_key::*;
pub use linear_transformation::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_key::*;
