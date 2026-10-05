use poulpy_core::{
    GetDistribution, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPublicKeyShareOwned;

/// Collective public key: every party generates a share under the common seed,
/// and any party aggregates the shares and finalizes the key of the ideal
/// secret, the sum of the parties' secrets.
pub trait GLWEPublicKeyMHEProtocol<BE: Backend> {
    fn mhe_glwe_public_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`: for each of the `rank` key
    /// entries, the body of an encryption of zero under `sk` whose mask is
    /// drawn from an entry seed derived from `seed`, and the distribution of
    /// `sk`.
    ///
    /// All parties contributing to one key use the same `seed`. Every new
    /// key-generation run needs a fresh seed, distinct from seeds used for
    /// other keys or protocols. `source_xe` must provide fresh private errors,
    /// independently seeded for each party and purpose; never replay its stream
    /// or initialize it from the public `seed`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_key_share_gen<S>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout, entry seeds and base secret distribution.
    /// Aggregation records the sum of their independent party counts.
    fn mhe_glwe_public_key_share_aggregate(&self, res: &mut GLWEPublicKeyShareOwned<BE>, a: &GLWEPublicKeyShareOwned<BE>);

    fn mhe_glwe_public_key_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical key `res` and tags it
    /// with the shares' base distribution, which `glwe_encrypt_pk` draws its
    /// ephemerals from, as core's key takes its secret's. The entries need
    /// distinct seeds, and finalization rejects shares whose entries share
    /// one: entries sharing a mask would give ciphertexts whose masks are rank
    /// 1 in the ephemerals. Derived encryption metadata retains the base law
    /// and party count of the collective secret, including after preparation.
    fn mhe_glwe_public_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEPublicKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyAtViewMut<BE> + GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos;
}
