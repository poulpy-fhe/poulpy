//! Secret-key and public-key encryption of ciphertexts and evaluation keys.
//!
//! This module provides traits and implementations for encrypting various
//! lattice-based cryptographic objects, including:
//!
//! - **Ciphertexts**: [`GLWEEncryptSk`], [`GLWEEncryptPk`], [`GGLWEEncryptSk`],
//!   [`GGSWEncryptSk`], [`LWEEncryptSk`] for encrypting plaintexts under
//!   GLWE, GGLWE, GGSW, and LWE schemes.
//!
//! - **Key-switching keys**: [`GLWESwitchingKeyEncryptSk`], [`LWESwitchingKeyEncrypt`],
//!   [`GLWEToLWESwitchingKeyEncryptSk`], [`LWEToGLWESwitchingKeyEncryptSk`] for
//!   generating keys that enable switching between different secret keys or
//!   between LWE and GLWE domains.
//!
//! - **Evaluation keys**: [`GLWEAutomorphismKeyEncryptSk`], [`GLWETensorKeyEncryptSk`],
//!   [`GGLWEToGGSWKeyEncryptSk`] for generating keys used in automorphism,
//!   tensor product, and GGLWE-to-GGSW conversion operations.
//!
//! - **Public keys**: [`GLWEPublicKeyGenerate`] for generating GLWE public keys
//!   from secret keys.
//!
//! Encryption methods follow a consistent pattern with PRNG sources:
//! - `source_xa`: source for mask/randomness sampling
//! - `source_xe`: source for error/noise sampling
//! - `source_xu`: source for uniform sampling (used in public-key encryption)
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
pub mod glwe_public_key;
pub mod glwe_switching_key;
pub mod glwe_tensor_key;
pub mod glwe_to_lwe_key;
pub mod lwe;
pub mod lwe_switching_key;
pub mod lwe_to_glwe_key;

pub use crate::api::{DeclaredK, EncryptionInfos, GGSWEncryptSk, GLWEEncryptSk};
pub use compressed::*;
pub use gglwe::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_public_key::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
pub use glwe_to_lwe_key::*;
pub use lwe::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_key::*;
use poulpy_hal::layouts::NoiseInfos;

use crate::layouts::{
    GGLWEInfos, GGLWELayout, GGSWInfos, GGSWLayout, GLWEInfos, GLWELayout, LWEInfos, LWELayout, TorusPrecision,
};
use anyhow::Result;

/// Standard deviation of the discrete Gaussian distribution used for error sampling
/// during encryption. Set to 3.2.
pub const DEFAULT_SIGMA_XE: f64 = 3.2;

/// Truncation bound for the discrete Gaussian error distribution, defined as 6.0 * [DEFAULT_SIGMA_XE].
/// Samples are rejected if their absolute value exceeds this bound.
pub const DEFAULT_BOUND_XE: f64 = 6.0 * DEFAULT_SIGMA_XE;

impl DeclaredK for LWELayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl DeclaredK for GLWELayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl DeclaredK for GGLWELayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl DeclaredK for GGSWLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

pub struct EncryptionLayout<L> {
    pub layout: L,
    pub noise: NoiseInfos,
}

impl<L: DeclaredK> EncryptionLayout<L> {
    pub fn new(layout: L, noise: NoiseInfos) -> Result<Self> {
        anyhow::ensure!(
            noise.k <= layout.max_k().as_usize(),
            "k_xe: {} > layout.max_k(): {}",
            noise.k,
            layout.max_k()
        );
        Ok(Self { layout, noise })
    }

    pub fn new_from_default_sigma(layout: L) -> Result<Self> {
        let noise = NoiseInfos::new(layout.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?;
        Self::new(layout, noise)
    }
}

impl<L> EncryptionInfos for EncryptionLayout<L> {
    fn noise_infos(&self) -> NoiseInfos {
        self.noise
    }
}

impl EncryptionInfos for NoiseInfos {
    fn noise_infos(&self) -> NoiseInfos {
        *self
    }
}

impl<L: LWEInfos> LWEInfos for EncryptionLayout<L> {
    fn base2k(&self) -> crate::layouts::Base2K {
        self.layout.base2k()
    }

    fn n(&self) -> crate::layouts::Degree {
        self.layout.n()
    }

    fn size(&self) -> usize {
        self.layout.size()
    }
}

impl<L: GLWEInfos> GLWEInfos for EncryptionLayout<L> {
    fn rank(&self) -> crate::layouts::Rank {
        self.layout.rank()
    }
}

impl<L: GGLWEInfos> GGLWEInfos for EncryptionLayout<L> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.layout.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.layout.dsize()
    }

    fn rank_in(&self) -> crate::layouts::Rank {
        self.layout.rank_in()
    }

    fn rank_out(&self) -> crate::layouts::Rank {
        self.layout.rank_out()
    }
}

impl<L: GGSWInfos> GGSWInfos for EncryptionLayout<L> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.layout.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.layout.dsize()
    }
}
