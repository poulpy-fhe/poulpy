use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{
        GLWECompressedSeed, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos,
        GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Collective public key: every party publishes a share under the common seed,
/// the shares are aggregated, and any party finalizes the key of the ideal
/// secret, the sum of the parties' secrets.
pub trait GLWEPublicKeyShare<BE: Backend> {
    fn glwe_public_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share of one public key entry into `res`: the body
    /// of an encryption of zero under `sk` whose mask is drawn from `seed`. A
    /// key of rank `r` has `r` entries, each shared under its own seed.
    ///
    /// All parties contributing to one entry use the same `seed`. Every
    /// new key-generation run needs a fresh seed, distinct from seeds used for
    /// other keys or protocols. `source_xe` must provide fresh private errors,
    /// independently seeded for each party and purpose; never replay its stream
    /// or initialize it from the public `seed`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares of every entry, in order, into `res` and
    /// tags it with `dist`, the distribution `glwe_encrypt_pk` draws its
    /// ephemerals from. The entries need distinct seeds: entries sharing a mask
    /// would give ciphertexts whose masks are rank 1 in the ephemerals.
    fn glwe_public_key_finalize<R, P>(&self, res: &mut R, pats: &[P], dist: Distribution, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos;
}
