use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};
use poulpy_core::{
    GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Encrypts coefficient-domain key bodies with deterministic per-element mask seeds.
pub trait BlindRotationKeyCompressedEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
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
    ///
    /// # Panics
    ///
    /// If the key count differs from the dimension of `sk_lwe`, the rank of
    /// `sk_glwe` from the key rank, or `sk_lwe` is not of distribution
    /// `BinaryBlock`, `BinaryFixed`, `BinaryProb` or `ZERO`; `res` and
    /// `source_xe` are then untouched.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1>(
        &self,
        res: &mut BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}
