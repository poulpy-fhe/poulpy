use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Normalized, ScratchArena},
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
/// `k_out = k` (the full encryption budget).
///
/// Errors with `InsufficientHomomorphicCapacity` if `k < pt.log_delta`
/// (i.e., the encryption key does not provide enough headroom for the
/// requested plaintext precision).
pub trait CKKSEncryptOps<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<Dct, Dpt, S, E: EncryptionInfos>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds;
}

/// Secret-key decryption of a CKKS ciphertext.
///
/// Recovers the plaintext polynomial from a CKKS ciphertext under the given
/// secret key.  The result is a [`CKKSPlaintext`](crate::layouts::CKKSPlaintext)
/// in the ZNX (torus) domain, ready for coefficient-domain or slot decoding.
///
/// # Metadata
///
/// The output frame is the **destination's preset** `(log_delta, log_budget)`:
/// the decrypted polynomial is shifted into `pt`'s frame, which lets a caller
/// extract at a precision different from the ciphertext's (see
/// `ckks_extract_pt_with_meta`). `pt`'s metadata is **not** modified by this
/// call — in particular it is *not* stamped from the ciphertext.
///
/// To decrypt "at the ciphertext's frame" (the common case), preset the
/// destination before calling:
///
/// ```text
/// pt.set_meta(ct.meta());        // log_delta_out  = ct.log_delta
///                                // log_budget_out = pt.max_k − ct.log_delta
/// ```
///
/// A freshly allocated plaintext has `log_delta = 0`, which decodes the raw
/// torus value as an integer frame — almost never what you want after CKKS
/// arithmetic; always preset the frame.
///
/// Errors with `PlaintextAlignmentImpossible` if the requested effective
/// precision `pt.log_delta + pt.log_budget` exceeds what the ciphertext can
/// supply (`ct.log_budget + pt.log_delta`), and with
/// `PlaintextBase2KMismatch` on differing `base2k`.
pub trait CKKSDecryptOps<BE: Backend> {
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds;
}
