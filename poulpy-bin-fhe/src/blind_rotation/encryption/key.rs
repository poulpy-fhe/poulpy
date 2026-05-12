use poulpy_hal::{
    layouts::{Backend, HostBackend, HostDataMut, ScratchArena},
    source::Source,
};

use poulpy_core::{
    EncryptionInfos, GetDistribution, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey};

/// Backend-level key-encryption trait for [`BlindRotationKey`].
///
/// Implemented for `Module<BE>` when the backend supports GGSW secret-key
/// encryption.  The [`BlindRotationKey::encrypt_sk`] convenience method
/// delegates to this trait.
///
/// Callers must supply:
/// - `sk_glwe`: The GLWE secret key (in prepared / DFT form) used to encrypt
///   each GGSW element.
/// - `sk_lwe`: The LWE secret key whose individual bits are encrypted.  Its
///   distribution must be one of `BinaryBlock`, `BinaryFixed`, `BinaryProb`,
///   or `ZERO` (debugging only).
/// - `source_xa`: Randomness source for GGSW mask components.
/// - `source_xe`: Randomness source for GGSW error components.
///
/// # Panics
///
/// Panics if `sk_lwe.dist()` is not a supported binary distribution.
pub trait BlindRotationKeyEncryptSk<BRA: BlindRotationAlgo, B: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_encrypt_sk`][Self::blind_rotation_key_encrypt_sk].
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Encrypts each bit of `sk_lwe` as a GGSW ciphertext under `sk_glwe`,
    /// storing the result in `res`.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_encrypt_sk<'s, S0, S1, E>(
        &self,
        res: &mut BlindRotationKey<B::OwnedBuf, BRA>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        S0: GLWESecretPreparedToBackendRef<B> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<B> + LWEInfos + GetDistribution,
        B: 's,
        ScratchArena<'s, B>: ScratchArenaTakeCore<'s, B>;
}

impl<D: HostDataMut, BRA: BlindRotationAlgo> BlindRotationKey<D, BRA> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<'s, M, S0, S1, E, BE>(
        &mut self,
        module: &M,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        E: EncryptionInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: Backend<OwnedBuf = D> + HostBackend + 's,
    {
        module.blind_rotation_key_encrypt_sk(self, sk_glwe, sk_lwe, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> BlindRotationKey<Vec<u8>, BRA> {
    pub fn encrypt_sk_tmp_bytes<A, M, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
    {
        module.blind_rotation_key_encrypt_sk_tmp_bytes(infos)
    }
}
