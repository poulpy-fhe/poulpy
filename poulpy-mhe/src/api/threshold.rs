use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{
        GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{
    GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned, GLWEShamirLayout, GLWEShamirPolynomialOwned, GLWEShamirShareOwned,
    GLWEWideSecretOwned, GLWEWideSecretPreparedOwned,
};

/// Shamir thresholdization over the Galois ring `GR(2^k, gr_degree)`: every
/// party shares its secret with a Shamir polynomial, sends its evaluation at
/// every party's point, and aggregates the shares it receives into a
/// t-out-of-N share of the sum of the parties' secrets; every party of an
/// active set of at least the threshold finalizes its share into an additive
/// share of the secret, an integer polynomial modulo `2^k`, and the active
/// parties' additive shares sum to the secret.
///
/// Party `i`, `1 <= i < 2^gr_degree`, has the point whose coefficients are the
/// bits of `i`.
///
/// Shares are secret: a party sends each one over a private channel, since
/// any `threshold` of them reveal the secret.
pub trait GLWEShamirMHEProtocol<BE: Backend> {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize;

    /// Writes into `res` a Shamir polynomial whose constant term is `sk` and
    /// whose other coefficients are uniform over the Galois ring, drawn from
    /// `source_xm`, which must stay secret to the party and never be replayed.
    fn mhe_glwe_shamir_polynomial_gen<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos;

    fn mhe_glwe_shamir_share_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize;

    /// Writes into `res` the evaluation of `poly` at the point of party
    /// `recipient`.
    fn mhe_glwe_shamir_share_gen(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_shamir_share_aggregate(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>);

    fn mhe_glwe_shamir_share_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res` the additive share of party `own` among `actives`.
    fn mhe_glwe_shamir_share_finalize(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Preparation of an additive share for key switching: every base-`2^base2k`
/// digit is prepared as a small secret.
pub trait GLWEWideSecretPrepare<BE: Backend> {
    fn mhe_glwe_wide_secret_prepare_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_wide_secret_prepare(
        &self,
        res: &mut GLWEWideSecretPreparedOwned<BE>,
        sk: &GLWEWideSecretOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Threshold key switching: the parties of an active set switch a ciphertext
/// with their combined additive shares in place of their secrets. The shares
/// aggregate and finalize as [`GLWEKeyswitchMHEProtocol`](crate::api::GLWEKeyswitchMHEProtocol)'s shares.
///
/// Switching to the zero secret is threshold decryption: the finalized
/// ciphertext's body is the plaintext plus noise.
pub trait GLWEThresholdKeyswitchMHEProtocol<BE: Backend> {
    /// `infos` is the ciphertext layout; the share has that layout at rank 0.
    fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res`, a rank-0 GLWE, the inner product of the masks of
    /// `ct` with `sk_in`, as precise as `ct`, minus their inner product with
    /// `sk_out`, plus smudging noise drawn with `flood`.
    ///
    /// `failure_bits` is a positive numerical failure target for the whole
    /// wide-secret product in this call, under the backend's `max_base2k`
    /// model. It is separate from the statistical security of `flood`.
    /// The reference rejects digit bases exceeding that budget, insufficient
    /// coefficient headroom for accumulation, and backends without a product
    /// model.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_keyswitch_share_gen<C, S, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        sk_out: &S,
        flood: &E,
        failure_bits: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos;

    /// Adds share `a` into `res`, which starts as the first share, as
    /// [`GLWEKeyswitchMHEProtocol`](crate::api::GLWEKeyswitchMHEProtocol) does.
    fn mhe_glwe_threshold_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        a: &GLWEKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes(&self) -> usize;

    /// Finalizes the aggregated shares into `res`, as
    /// [`GLWEKeyswitchMHEProtocol`](crate::api::GLWEKeyswitchMHEProtocol) does.
    fn mhe_glwe_threshold_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Threshold key switching to a public key. The shares aggregate and finalize
/// as [`GLWEPublicKeyswitchMHEProtocol`](crate::api::GLWEPublicKeyswitchMHEProtocol)'s shares.
pub trait GLWEThresholdPublicKeyswitchMHEProtocol<BE: Backend> {
    /// `ct_infos` is the ciphertext layout, `res_infos` the share layout and
    /// `pk_infos` the public key layout.
    fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes<A, B, P>(
        &self,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    /// Writes into `res` the encryption under `pk_out` of the inner product
    /// of the masks of `ct` with `sk_in`, plus smudging noise drawn with
    /// `flood`.
    ///
    /// `failure_bits` has the same numerical meaning and budget checks as in
    /// [`GLWEThresholdKeyswitchMHEProtocol::mhe_glwe_threshold_keyswitch_share_gen`]. It
    /// covers the wide-secret product, not the public-key encryption.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_public_keyswitch_share_gen<C, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        failure_bits: usize,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share, as
    /// [`GLWEPublicKeyswitchMHEProtocol`](crate::api::GLWEPublicKeyswitchMHEProtocol) does.
    fn mhe_glwe_threshold_public_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes(&self) -> usize;

    /// Finalizes the aggregated shares into `res`, as
    /// [`GLWEPublicKeyswitchMHEProtocol`](crate::api::GLWEPublicKeyswitchMHEProtocol) does.
    fn mhe_glwe_threshold_public_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}
