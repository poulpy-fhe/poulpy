use crate::layouts::{GLWEPrivateKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    Noise,
    layouts::{
        GLWEInfos, GLWEMaskToBackendRef, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Collective key switching to a secret key: every party publishes a share,
/// the shares are aggregated with `mhe_glwe_private_keyswitch_share_aggregate`, and any party
/// finalizes the ciphertext under the ideal output secret, the sum of the
/// parties' output secrets.
///
/// A share is a rank-0 core `GLWE`, the party's part of the new body: the
/// inner product of the mask of `ct` with `sk_in - sk_out`, plus its `flood`,
/// drawn from `source_smudge` at the share's precision. Share generation reads
/// the mask alone, a core `GLWEMask` or the mask of the `GLWE` itself, which
/// must be canonical. The inner product is exact, so the flood alone hides the
/// secrets; size it as the smudging section of `docs/mhe-contracts.md`
/// describes. Aggregation adds the shares; finalization adds them to the body
/// of `ct` and normalizes.
///
/// The mask must be the session's honest ciphertext's: a crafted mask, such as a
/// large constant in one column, reveals that component of every party's
/// secret above the flood.
pub trait GLWEPrivateKeyswitchMHEProtocol<BE: Backend> {
    /// `infos` is the layout of the ciphertext or its mask; the share has that layout at rank 0.
    fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`, a rank-0 GLWE: the inner product
    /// of `mask`, the mask of the ciphertext, with `sk_in - sk_out`, plus
    /// smudging noise drawn with `flood`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_private_keyswitch_share_gen<C, S1, S2>(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: Noise,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_private_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        a: &GLWEPrivateKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res` the ciphertext `ct` with the aggregated shares added
    /// to its body.
    fn mhe_glwe_private_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPrivateKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Collective key switching to a public key: the finalized ciphertext is
/// encrypted under `pk_out`, at its rank.
///
/// A share is a core `GLWE` at the rank of `pk_out`: the public-key encryption
/// of the inner product of the mask of `ct` with `sk_in`, whose body error is
/// the party's `flood`, drawn from `source_smudge` at the share's precision.
/// The mask columns carry the encryption errors from `source_xe`, and the
/// ephemerals come from `source_xu`. Share generation reads the mask alone, as
/// for [`GLWEPrivateKeyswitchMHEProtocol`], and the flood is sized as there.
/// `pk_out` must be at least as precise as the share. Aggregation adds the
/// shares; finalization normalizes their sum and adds the body of `ct`, which
/// must be honest as for [`GLWEPrivateKeyswitchMHEProtocol`].
pub trait GLWEPublicKeyswitchMHEProtocol<BE: Backend> {
    /// `ct_infos` is the layout of the ciphertext or its mask, `res_infos` the share layout and
    /// `pk_infos` the public key layout.
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    /// Writes this party's share into `res`: the encryption under `pk_out` of
    /// the inner product of `mask`, the mask of the ciphertext, with `sk_in`,
    /// with smudging noise drawn with `flood` as its body error.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_keyswitch_share_gen<C, S, K>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_public_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res`, laid out as the share, the aggregated shares with
    /// the body of `ct` added to their body.
    fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}
