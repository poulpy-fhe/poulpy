use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPatCompressedOwned;

/// Collective public key: every party publishes a share under the common seed,
/// the shares are aggregated, and any party finalizes the key of the ideal
/// secret, the sum of the parties' secrets.
pub trait GLWEPublicKeyShare<BE: Backend> {
    fn glwe_public_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`: the body of an encryption of zero
    /// under `sk` whose mask is drawn from `seed`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share<S, E>(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into `res` and tags it with `dist`, the
    /// distribution `glwe_encrypt_pk` draws its ephemeral secret from.
    fn glwe_public_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<BE>,
        dist: Distribution,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos;
}
