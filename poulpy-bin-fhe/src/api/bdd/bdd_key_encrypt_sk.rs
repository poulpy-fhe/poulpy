use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
/// Backend-level factory for encrypting a [`BDDKey`] under a secret key.
///
/// Implemented for `Module<BE>` when the backend supports circuit-bootstrapping
/// and switching-key encryption.  Callers should prefer the convenience method
/// [`BDDKey::encrypt_sk`].
pub trait BDDKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by
    /// [`bdd_key_encrypt_sk`][Self::bdd_key_encrypt_sk].
    /// Fills `res` with key material encrypted under `sk_lwe` / `sk_glwe`.
    ///
    /// `source_xa` supplies mask randomness; `source_xe` supplies error
    /// randomness.  The scratch arena must be at least
    /// [`bdd_key_encrypt_sk_tmp_bytes`][Self::bdd_key_encrypt_sk_tmp_bytes]
    /// bytes.
    ///
    /// When `res.ks_glwe` is `Some`, a fresh intermediate GLWE key is sampled
    /// from `source_xe` and used as the bridging secret; `ks_lwe` is then
    /// encrypted under that intermediate key rather than `sk_glwe` directly.
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}
