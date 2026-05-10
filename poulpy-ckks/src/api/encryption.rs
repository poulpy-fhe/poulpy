use anyhow::Result;
use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Secret-key encryption of a CKKS plaintext.
///
/// Encrypts a [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) under the
/// given secret key.  The plaintext must already be encoded (coefficient
/// domain or slot domain) at the desired precision.
///
/// # Metadata
///
/// The encryption parameters supply the total torus budget `k` via
/// `enc_infos.noise_infos().k`.  The ciphertext metadata is set to:
///
/// ```text
/// log_delta_out  = pt.log_delta
/// log_budget_out = k − pt.log_delta
/// ```
///
/// `effective_k_out = k` (the full encryption budget).
///
/// Errors with `InsufficientHomomorphicCapacity` if `k < pt.log_delta`
/// (i.e., the encryption key does not provide enough headroom for the
/// requested plaintext precision).
pub trait CKKSEncrypt<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Dct, Dpt, S, E: EncryptionInfos>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSCtBounds,
        BE: 's;
}

/// Secret-key decryption of a CKKS ciphertext.
///
/// Recovers the plaintext polynomial from a CKKS ciphertext under the given
/// secret key.  The result is a [`CKKSPlaintext`](crate::layouts::CKKSPlaintext)
/// in the ZNX (torus) domain, ready for coefficient-domain or slot decoding.
///
/// # Metadata
///
/// The output plaintext inherits the ciphertext's metadata unchanged:
///
/// ```text
/// log_delta_out  = ct.log_delta
/// log_budget_out = ct.log_budget
/// ```
pub trait CKKSDecrypt<BE: Backend> {
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds;
}
