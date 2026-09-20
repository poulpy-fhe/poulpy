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

pub use crate::api::{EncryptionInfos, GGSWEncryptSk, GLWEEncryptSk, GLWEMaskFill, LWEFillMask};
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

use crate::layouts::{GGLWEInfos, GGSWInfos, GLWEInfos, LWEInfos, TorusPrecision};
use anyhow::Result;

/// Standard deviation of the discrete Gaussian distribution used for error sampling
/// during encryption. Set to 3.2.
pub const DEFAULT_SIGMA_XE: f64 = 3.2;

/// Truncation bound for the discrete Gaussian error distribution, defined as 6.0 * [DEFAULT_SIGMA_XE].
/// Samples are rejected if their absolute value exceeds this bound.
pub const DEFAULT_BOUND_XE: f64 = 6.0 * DEFAULT_SIGMA_XE;

/// Parameters of the discrete Gaussian error added at torus precision `2^-k`.
///
/// The descriptor [`VecZnxAddNormal`](crate::VecZnxAddNormal) and
/// [`VecZnxBigAddNormal`](crate::VecZnxBigAddNormal) take: the backend draws
/// the error in place through [`SamplingImpl`](crate::oep::SamplingImpl).
#[derive(Clone, Copy, Debug)]
pub struct NoiseInfos {
    pub k: usize,
    pub sigma: f64,
    pub bound: f64,
}

impl NoiseInfos {
    pub fn new(k: usize, sigma: f64, bound: f64) -> Result<Self> {
        anyhow::ensure!(sigma.is_sign_positive(), "sigma must be positive");
        anyhow::ensure!(sigma >= 1.0, "sigma must be greater or equal to 1");
        anyhow::ensure!(bound >= sigma, "bound: {bound} must be greater or equal to sigma: {sigma}");
        Ok(Self { k, sigma, bound })
    }

    /// Target limb and the number of unused low bits it holds.
    pub fn target_limb_and_shift(&self, base2k: usize) -> (usize, u32) {
        let limb: usize = self.k.div_ceil(base2k) - 1;
        (limb, ((limb + 1) * base2k - self.k) as u32)
    }
}

#[derive(Debug)]
pub struct EncryptionLayout<L> {
    pub layout: L,
    pub noise: NoiseInfos,
}

impl<L: LWEInfos> EncryptionLayout<L> {
    pub fn new(layout: L, noise: NoiseInfos) -> Result<Self> {
        anyhow::ensure!(
            noise.k <= layout.k().as_usize(),
            "k_xe: {} > layout.k(): {}",
            noise.k,
            layout.k()
        );
        Ok(Self { layout, noise })
    }

    pub fn new_from_default_sigma(layout: L) -> Result<Self> {
        // Place the error at the object's full precision `k` (the physical
        // bottom limb). For key layouts `k()` already returns the total
        // `dnum*dsize*base2k + k_aux`, so the guard region is properly encrypted.
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

    fn max_size(&self) -> usize {
        self.layout.max_size()
    }

    fn size(&self) -> usize {
        self.layout.size()
    }

    fn k(&self) -> TorusPrecision {
        self.layout.k()
    }
}

impl<L: GLWEInfos> GLWEInfos for EncryptionLayout<L> {
    fn rank(&self) -> crate::layouts::Rank {
        self.layout.rank()
    }
}

impl<L: GGLWEInfos> GGLWEInfos for EncryptionLayout<L> {
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.layout.k_aux()
    }

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
    fn k_aux(&self) -> crate::layouts::TorusPrecision {
        self.layout.k_aux()
    }

    fn dnum(&self) -> crate::layouts::Dnum {
        self.layout.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.layout.dsize()
    }
}
