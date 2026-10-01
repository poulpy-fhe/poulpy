use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWESecretToBackendRef, prepared::GGLWEPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

use crate::layouts::GGSWShareOwned;

/// Collective GGSW: parties holding secrets `s_i` and messages `m_i` produce, in
/// one round, the GGSW of `Sum_i m_i` under the ideal secret `s = Sum_i s_i`.
///
/// Column `j >= 1` encrypts `m * s_j`, which no party can encrypt alone: every
/// party also holds an ephemeral secret `u_i` (rank 1), used in two places:
/// as the `u` of every GGSW share it generates, and as the input secret of its
/// share of the collective switching key from `u = Sum_i u_i` to `s`
/// ([`GLWESwitchingKeyMHEProtocol`](crate::api::GLWESwitchingKeyMHEProtocol)).
/// That key, the ephemeral key, is what finalization consumes; one serves every
/// GGSW of a key set, so `u_i` is reused across shares, never drawn per share.
///
/// The ephemeral secret must be freshly sampled, independent of the party's
/// secret, and kept as private as it: with `u_i = s_i`, the two halves of a
/// column reveal the message.
///
/// Every GGSW needs its own seed, and the ephemeral key a seed distinct from
/// all of them: two shares over the same masks and the same `u_i` reveal the
/// difference of their messages.
pub trait GGSWMHEProtocol<BE: Backend> {
    fn mhe_ggsw_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Writes this party's share of the GGSW of `pt` into `res`: column 0 as
    /// the bodies of a seeded encryption of `pt` under `sk`, and for every
    /// column `j >= 1` the bodies of seeded encryptions of `pt` under `u` and of
    /// zero under the component `j` of `sk`, over common masks. `u` is the
    /// ephemeral secret this party's ephemeral key share was generated from.
    #[allow(clippy::too_many_arguments)]
    fn mhe_ggsw_share_gen<P, S, U, E>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        U: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout and seeds.
    fn mhe_ggsw_share_aggregate(&self, res: &mut GGSWShareOwned<BE>, a: &GGSWShareOwned<BE>);

    /// `res_infos` is the GGSW layout, `key_infos` the ephemeral key layout.
    fn mhe_ggsw_share_finalize_tmp_bytes<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos;

    /// Expands the aggregated shares into the canonical `res`, a GGSW at the share's
    /// layout: column 0 from the seeds, column `j >= 1` as the key switch of
    /// the negated second half from `u` to `s` under `key`, plus the first half
    /// in mask column `j`. `key` is the prepared ephemeral key; its gadget
    /// (`dnum * dsize * base2k`) must cover the GGSW precision.
    fn mhe_ggsw_share_finalize<R, K>(&self, res: &mut R, share: &GGSWShareOwned<BE>, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}
