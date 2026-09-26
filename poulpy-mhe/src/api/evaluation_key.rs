use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned};

/// Collective GLWE switching key: every party publishes a share under the
/// common seed, the shares are aggregated, and any party finalizes the key
/// switching from the sum of the input secrets to the sum of the output secrets.
pub trait GLWESwitchingKeyShare<BE: Backend> {
    fn glwe_switching_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Writes this party's share into `res`: the bodies of an encryption of
    /// `sk_in` under `sk_out` whose masks are drawn from `seed`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_share<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The degrees
    /// must match.
    fn glwe_switching_key_share_aggregate_assign(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        a: &GLWESwitchingKeyPatCompressedOwned<BE>,
    );

    fn glwe_switching_key_share_normalize_tmp_bytes(&self) -> usize;

    fn glwe_switching_key_share_normalize_assign(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn glwe_switching_key_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into `res` and copies their degrees.
    fn glwe_switching_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWESwitchingKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut;
}

/// Collective GLWE automorphism key for one Galois element `p`: the finalized
/// key maps `X -> X^p` under the ideal secret, the sum of the parties' secrets.
pub trait GLWEAutomorphismKeyShare<BE: Backend> {
    fn glwe_automorphism_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Writes this party's share into `res`: the bodies of the automorphism key
    /// encryption of `sk` for `p` whose masks are drawn from `seed`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_share<S, E>(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The Galois
    /// elements must match.
    fn glwe_automorphism_key_share_aggregate_assign(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
        a: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
    );

    fn glwe_automorphism_key_share_normalize_tmp_bytes(&self) -> usize;

    fn glwe_automorphism_key_share_normalize_assign(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn glwe_automorphism_key_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into `res` and copies their Galois element.
    fn glwe_automorphism_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement;
}
