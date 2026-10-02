use poulpy_core::{
    EncryptionInfos,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos,
        prepared::{GGLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
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
/// party also holds an ephemeral secret `u_i` of the GGSW's rank, used in two
/// places: as the `u` of every GGSW share it generates, and as the input secret
/// of its share of the collective switching key from `u = Sum_i u_i` to `s`
/// ([`GLWESwitchingKeyMHEProtocol`](crate::api::GLWESwitchingKeyMHEProtocol)).
/// That key, the ephemeral key, is what finalization consumes; one serves every
/// GGSW of a key set, so `u_i` is reused across shares, never drawn per share.
///
/// Every gadget row of column `j` draws a common `r x r` mask matrix `A` from
/// the seed. A party publishes the bodies of `r` encryptions of zero under
/// `s_i` over the rows of `A`, and of `r` encryptions under `u_i` over its
/// columns, the one of column `j` carrying the message. The key switch of the
/// first from `u` to `s` cancels the cross term `Sum_i u_i <A_i, s>` of the
/// second, so every published body is a rank-`r` encryption under the whole of
/// `s_i` or `u_i`, never under a single component.
///
/// The ephemeral secret must be freshly sampled, independent of the party's
/// secret, and kept as private as it: with `u_i = s_i`, the two halves of a
/// column reveal the message.
///
/// Every GGSW needs its own seed, and the ephemeral key a seed distinct from
/// all of them: two shares over the same masks and the same `u_i` reveal the
/// difference of their messages.
///
/// A key set that already holds the collective tensor key
/// ([`GLWETensorKeyMHEProtocol`](crate::api::GLWETensorKeyMHEProtocol)) can
/// instead build the GGSW from a collective GGLWE of `m` alone: every party
/// shares the seeded encryption of `m_i` under `s_i` (column 0 of this share),
/// and [`GGSWFromGGLWE`](poulpy_core::GGSWFromGGLWE) expands the finalized
/// GGLWE with the tensor key, laid out as a
/// [`GGLWEToGGSWKey`](poulpy_core::layouts::GGLWEToGGSWKey), whose key `i`
/// column `j` is the tensor key entry of `s_i * s_j`.
pub trait GGSWMHEProtocol<BE: Backend> {
    fn mhe_ggsw_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Writes this party's share of the GGSW of `pt` into `res`: column 0 as
    /// the bodies of a seeded encryption of `pt` under `sk`, and for every
    /// column `j >= 1` the bodies of the encryptions of zero under `sk` over the
    /// rows of the common mask matrices and of `pt` under `u` over their
    /// columns. `u` is the ephemeral secret, of `sk`'s rank, this party's
    /// ephemeral key share was generated from.
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
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
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
    /// layout: column 0 from the seeds, column `j >= 1` as the key switch from
    /// `u` to `s` under `key` of the negated zero-encryption bodies in the mask
    /// columns, plus the message-encryption bodies in the mask columns. Column
    /// 0's masks come from the seeds, so only its bodies are normalized. `key`
    /// is the prepared ephemeral key, from the GGSW's rank to itself; its gadget
    /// (`dnum * dsize * base2k`) must cover the GGSW precision, and one guard
    /// digit (`k_aux >= base2k + log2 n`) keeps its noise far below the
    /// circular term.
    fn mhe_ggsw_share_finalize<R, K>(&self, res: &mut R, share: &GGSWShareOwned<BE>, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}
