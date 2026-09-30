use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPublicKeyShareOwned;

/// Collective public key: every party generates a share under the common seed,
/// and any party aggregates the shares and finalizes the key of the ideal
/// secret, the sum of the parties' secrets.
pub trait GLWEPublicKeyProtocol<BE: Backend> {
    fn glwe_public_key_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`: for each of the `rank` key
    /// entries, the body of an encryption of zero under `sk` whose mask is
    /// drawn from an entry seed derived from `seed`.
    ///
    /// All parties contributing to one key use the same `seed`. Every new
    /// key-generation run needs a fresh seed, distinct from seeds used for
    /// other keys or protocols. `source_xe` must provide fresh private errors,
    /// independently seeded for each party and purpose; never replay its stream
    /// or initialize it from the public `seed`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_gen<S, E>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout and entry seeds.
    fn glwe_public_key_aggregate(&self, res: &mut GLWEPublicKeyShareOwned<BE>, a: &GLWEPublicKeyShareOwned<BE>);

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical key `res` and tags it
    /// with `dist`, the distribution `glwe_encrypt_pk` draws its ephemerals
    /// from. The entries need distinct seeds: entries sharing a mask would give
    /// ciphertexts whose masks are rank 1 in the ephemerals.
    fn glwe_public_key_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEPublicKeyShareOwned<BE>,
        dist: Distribution,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos;
}
