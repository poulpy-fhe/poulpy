use crate::layouts::{GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Collective key switching to a secret key: every party publishes a share,
/// the shares are aggregated with core `glwe_add_assign`, and any party
/// finalizes the ciphertext under the ideal output secret, the sum of the
/// parties' output secrets.
///
/// `flood` supplies [`poulpy_core::SmudgingNoise`] through [`SmudgingInfos`].
/// The Gaussian option has integer scale `2^log_sigma` and an explicit cutoff;
/// the uniform option has a contiguous `bits`-bit integer support. Both scale
/// the complete integer by `2^-k`, including low bits beyond one machine word.
/// Use the sampling destination's precision for `k` to hide fine input errors.
///
/// Each party needs its own statistical hiding margin. For Gaussian flooding,
/// the scale must dominate input encryption, evaluation and rounding error by
/// at least `2^lambda`, accounting for dimension and the whole transcript.
/// For an integer discrepancy vector `e`, the untruncated Gaussian's shift
/// distance is at most `||e||_2 / (2 sigma)`; uniform's is at most
/// `||e||_1 / 2^bits`. Budget Gaussian truncation separately. The API does not
/// infer input noise or `lambda`. Sum all flood bounds within the decoding
/// margin. Private error streams must be independently seeded and never replayed.
pub trait GLWEKeyswitchMHEProtocol<BE: Backend> {
    /// `infos` is the ciphertext layout; the share has that layout at rank 0.
    fn mhe_glwe_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`, a rank-0 GLWE: the inner product
    /// of the masks of `ct` with `sk_in - sk_out`, plus smudging noise drawn
    /// with `flood`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_keyswitch_share_gen<C, S1, S2, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_keyswitch_share_aggregate(&self, res: &mut GLWEKeyswitchShareOwned<BE>, a: &GLWEKeyswitchShareOwned<BE>);

    fn mhe_glwe_keyswitch_share_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res` the ciphertext `ct` with the aggregated shares added
    /// to its body.
    fn mhe_glwe_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Collective key switching to a public key: the finalized ciphertext is
/// encrypted under `pk_out`, at its rank. The smudging requirement is the
/// same as [`GLWEKeyswitchMHEProtocol`]; ordinary public-key encryption noise does
/// not replace the flood. `source_xu` and `source_xe` are independent private
/// streams, consumed without replay.
pub trait GLWEPublicKeyswitchMHEProtocol<BE: Backend> {
    /// `ct_infos` is the ciphertext layout, `res_infos` the share layout and
    /// `pk_infos` the public key layout.
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    /// Writes this party's share into `res`: the encryption under `pk_out` of
    /// the inner product of the masks of `ct` with `sk_in`, plus smudging
    /// noise drawn with `flood`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_keyswitch_share_gen<C, S, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

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
