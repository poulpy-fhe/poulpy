use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey};
use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Encrypts coefficient-domain key elements through the selected backend.
pub trait BlindRotationKeyEncryptSk<BRA: BlindRotationAlgo, B: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_encrypt_sk`][Self::blind_rotation_key_encrypt_sk].
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Encrypts each bit of `sk_lwe` as a GGSW ciphertext under `sk_glwe`,
    /// storing the result in `res`.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_encrypt_sk<S0, S1, E>(
        &self,
        res: &mut BlindRotationKey<B::OwnedBuf, BRA, B::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, B>,
    ) where
        S0: GLWESecretPreparedToBackendRef<B> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<B> + LWEInfos + GetDistribution;
}
