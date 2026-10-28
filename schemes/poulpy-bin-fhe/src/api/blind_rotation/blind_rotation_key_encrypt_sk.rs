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
    ///
    /// `sk_lwe` is the LWE secret whose bits are encrypted, of distribution
    /// `BinaryBlock`, `BinaryFixed`, `BinaryProb` or `ZERO` (debugging only),
    /// `sk_glwe` the prepared GLWE secret of each GGSW; `source_xe` draws the
    /// errors and `source_xa` the masks.
    ///
    /// # Panics
    ///
    /// If the key count differs from the dimension of `sk_lwe`, the rank of
    /// `sk_glwe` from the key rank, or `sk_lwe` has another distribution;
    /// `res` and the sources are then untouched.
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
