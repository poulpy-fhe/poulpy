use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey};
use poulpy_core::{
    GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Encrypts coefficient-domain key elements through the selected backend.
pub trait BlindRotationKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_encrypt_sk`][Self::blind_rotation_key_encrypt_sk].
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Encrypts each bit of `sk_lwe` as a GGSW ciphertext under `sk_glwe`,
    /// storing the result in `res`.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}
