use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned};

/// Collective GLWE switching key: every party publishes a share under the
/// common seed, and any party aggregates and finalizes the shares with
/// [`GLWESwitchingKeyPatCompressedOps`](crate::api::GLWESwitchingKeyPatCompressedOps)
/// into the key switching from the sum of the input secrets to the sum of the
/// output secrets.
pub trait GLWESwitchingKeyShare<BE: Backend> {
    fn glwe_switching_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
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
}

/// Collective GLWE automorphism key for one Galois element `p`: the shares,
/// aggregated and finalized with
/// [`GLWEAutomorphismKeyPatCompressedOps`](crate::api::GLWEAutomorphismKeyPatCompressedOps),
/// give the key mapping `X -> X^p` under the ideal secret, the sum of the
/// parties' secrets.
pub trait GLWEAutomorphismKeyShare<BE: Backend> {
    fn glwe_automorphism_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
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
}
