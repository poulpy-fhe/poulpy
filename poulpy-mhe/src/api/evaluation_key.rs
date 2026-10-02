use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEAutomorphismKeyShareOwned, GLWESwitchingKeyShareOwned};

/// Collective GLWE switching key: every party generates a share under the
/// common seed, and any party aggregates the shares and finalizes the key
/// switching from the sum of the input secrets to the sum of the output secrets.
pub trait GLWESwitchingKeyMHEProtocol<BE: Backend> {
    fn mhe_glwe_switching_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Writes this party's share into `res`: the bodies of an encryption of
    /// `sk_in` under `sk_out` whose masks are drawn from `seed`.
    ///
    /// All parties contributing to one switching key use the same `seed`.
    /// Every new key-generation run needs a fresh seed, distinct from seeds
    /// used for other keys or protocols. Reusing masks for different input
    /// secrets under one output secret reveals their gadget-scaled difference
    /// plus small error. `source_xe` must provide fresh private errors,
    /// independently seeded for each party and purpose; never replay its stream
    /// or initialize it from the public `seed`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_switching_key_share_gen<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyShareOwned<BE>,
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

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout, seed and degrees.
    fn mhe_glwe_switching_key_share_aggregate(
        &self,
        res: &mut GLWESwitchingKeyShareOwned<BE>,
        a: &GLWESwitchingKeyShareOwned<BE>,
    );

    fn mhe_glwe_switching_key_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical key `res` and copies
    /// their degrees. `res` must have the share's layout.
    fn mhe_glwe_switching_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWESwitchingKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut;
}

/// Collective GLWE automorphism key for one Galois element `p`: every party
/// generates a share under the common seed, and any party aggregates the
/// shares and finalizes the key mapping `X -> X^p` under the ideal secret, the
/// sum of the parties' secrets.
pub trait GLWEAutomorphismKeyMHEProtocol<BE: Backend> {
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Writes this party's share into `res`: the bodies of the automorphism key
    /// encryption of `sk` for `p` whose masks are drawn from `seed`.
    ///
    /// All parties contributing to one automorphism key use the same `seed`.
    /// Every new key-generation run needs a fresh seed, distinct from seeds
    /// used for other keys or protocols, including other Galois elements in the
    /// same key set. `source_xe` must provide fresh private errors, independently
    /// seeded for each party and purpose; never replay its stream or initialize
    /// it from the public `seed`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_automorphism_key_share_gen<S, E>(
        &self,
        res: &mut GLWEAutomorphismKeyShareOwned<BE>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout, seed and Galois element.
    fn mhe_glwe_automorphism_key_share_aggregate(
        &self,
        res: &mut GLWEAutomorphismKeyShareOwned<BE>,
        a: &GLWEAutomorphismKeyShareOwned<BE>,
    );

    fn mhe_glwe_automorphism_key_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical key `res` and copies
    /// their Galois element. `res` must have the share's layout.
    fn mhe_glwe_automorphism_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEAutomorphismKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement;
}
