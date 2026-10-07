//! Secret-key and public-key encryption of ciphertexts and evaluation keys.
//!
//! This module provides traits and implementations for encrypting various
//! lattice-based cryptographic objects, including:
//!
//! - **Ciphertexts**: [`GLWEEncryptSk`], [`GLWEEncryptPk`](crate::api::GLWEEncryptPk), [`GGLWEEncryptSk`],
//!   [`GGSWEncryptSk`], [`LWEEncryptSk`](crate::api::LWEEncryptSk) for encrypting plaintexts under
//!   GLWE, GGLWE, GGSW, and LWE schemes.
//!
//! - **Key-switching keys**: [`GLWESwitchingKeyEncryptSk`], [`LWESwitchingKeyEncrypt`](crate::api::LWESwitchingKeyEncrypt),
//!   [`GLWEToLWESwitchingKeyEncryptSk`](crate::api::GLWEToLWESwitchingKeyEncryptSk), [`LWEToGLWESwitchingKeyEncryptSk`](crate::api::LWEToGLWESwitchingKeyEncryptSk) for
//!   generating keys that enable switching between different secret keys or
//!   between LWE and GLWE domains.
//!
//! - **Evaluation keys**: [`GLWEAutomorphismKeyEncryptSk`](crate::api::GLWEAutomorphismKeyEncryptSk), [`GLWETensorKeyEncryptSk`](crate::api::GLWETensorKeyEncryptSk),
//!   [`GGLWEToGGSWKeyEncryptSk`](crate::api::GGLWEToGGSWKeyEncryptSk) for generating keys used in automorphism,
//!   tensor product, and GGLWE-to-GGSW conversion operations.
//!
//! - **Public keys**: [`GLWEPublicKeyGenerate`](crate::api::GLWEPublicKeyGenerate) for generating GLWE public keys
//!   from secret keys.
//!
//! Encryption methods follow a consistent pattern with PRNG sources:
//! - `source_xa`: source for mask/randomness sampling
//! - `source_xe`: source for error/noise sampling
//! - `source_xu`: source for the public-key encryption ephemerals, drawn under the key's distribution
//!
//! Scratch space requirements for each operation can be queried via companion
//! `*_tmp_bytes` methods.

#![allow(clippy::too_many_arguments)]

pub mod compressed;
pub mod gglwe;
pub mod gglwe_to_ggsw_key;
pub mod ggsw;
pub mod glwe;
pub mod glwe_automorphism_key;
pub mod glwe_switching_key;
pub mod glwe_to_lwe_key;
pub mod lwe;
pub mod lwe_switching_key;
pub mod lwe_to_glwe_key;

pub use crate::api::{GGSWEncryptSk, GLWEEncryptSk, GLWEMaskFill, LWEFillMask};
pub use compressed::*;
pub use gglwe::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_switching_key::*;
pub use glwe_to_lwe_key::*;
pub use lwe::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_key::*;

/// Standard deviation of the discrete Gaussian distribution used for error sampling
/// during encryption. Set to 3.2.
pub const DEFAULT_SIGMA_XE: f64 = match crate::Noise::ENCRYPTION {
    crate::Noise::Gaussian { sigma } => sigma,
    crate::Noise::Uniform { .. } => unreachable!(),
};

/// Maximum absolute sample of the default encryption error distribution.
pub const DEFAULT_BOUND_XE: f64 = match crate::Noise::ENCRYPTION {
    crate::Noise::Gaussian { sigma } => (sigma * crate::Noise::CUTOFF_FACTOR as f64) as u64 as f64,
    crate::Noise::Uniform { .. } => unreachable!(),
};
