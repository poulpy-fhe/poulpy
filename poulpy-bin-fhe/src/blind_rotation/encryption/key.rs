use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    layouts::{Backend, Data, ScratchArena, ZnxWord},
    source::Source,
};

use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey};

/// Backend-level key-encryption trait for [`BlindRotationKey`].
///
/// Dispatched by `Module<BE>` through [`crate::oep::BlindRotationKeyEncryptSkImpl`].  The [`BlindRotationKey::encrypt_sk`] convenience method
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
pub use crate::api::BlindRotationKeyEncryptSk;

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKey<D, BRA, W> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S0, S1, E, BE>(
        &mut self,
        module: &M,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        E: EncryptionInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        module.blind_rotation_key_encrypt_sk(self, sk_glwe, sk_lwe, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> BlindRotationKey<AlignedBuf, BRA, i64> {
    pub fn encrypt_sk_tmp_bytes<A, M, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
    {
        module.blind_rotation_key_encrypt_sk_tmp_bytes(infos)
    }
}
