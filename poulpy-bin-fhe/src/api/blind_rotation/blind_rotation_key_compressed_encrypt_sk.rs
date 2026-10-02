use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};
use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Encrypts coefficient-domain key bodies with deterministic per-element mask seeds.
pub trait BlindRotationKeyCompressedEncryptSk<B: Backend, BRA: BlindRotationAlgo> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_compressed_encrypt_sk`][Self::blind_rotation_key_compressed_encrypt_sk].
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Encrypts each bit of `sk_lwe` as a compressed GGSW ciphertext under
    /// `sk_glwe`, storing the result in `res`.
    ///
    /// `seed_xa` is the 32-byte root seed from which per-element mask seeds
    /// are derived.  `source_xe` provides randomness for the error components.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1, E>(
        &self,
        res: &mut BlindRotationKeyCompressed<B::OwnedBuf, BRA, B::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, B>,
    ) where
        S0: GLWESecretPreparedToBackendRef<B> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<B> + LWEInfos + GetDistribution;
}
