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

/// Writes the plaintext of gadget row `row`, `pt[pt_col] * 2^-((row + 1) * k_digit)`,
/// into `res[res_col]`, canonical at radix `base2k` and precision `k`.
///
/// The scalar is read as a one-limb vector at radix 1, worth `pt / 2`, and
/// scaled by `2^(1 - (row + 1) * k_digit)`, so the row lands on any bit and not
/// only on a limb boundary. Every limb of `res[res_col]` is written. For
/// `k_digit = dsize * base2k` this is the scalar added to limb
/// `(dsize - 1) + row * dsize` and normalized, bit for bit.
///
/// Panics unless the row lies within the precision, `(row + 1) * k_digit <= k`.
#[allow(clippy::too_many_arguments)]
pub fn gadget_row_plaintext<BE, M>(
    module: &M,
    res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
    res_col: usize,
    base2k: usize,
    k: usize,
    row: usize,
    k_digit: usize,
    pt: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, BE>,
    pt_col: usize,
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) where
    BE: poulpy_hal::layouts::Backend,
    M: poulpy_hal::api::VecZnxNormalize<BE>,
{
    let position: usize = (row + 1) * k_digit;
    assert!(position <= k, "gadget row {row} at bit {position} is below the precision {k}");
    let src = poulpy_hal::layouts::scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(pt);
    module.vec_znx_normalize(res, base2k, k, 1 - position as i64, res_col, &src, 1, pt_col, scratch);
}
