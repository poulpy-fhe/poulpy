use poulpy_core::{
    EncryptionInfos, SmudgingNoise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEEncToShareShareOwned, GLWEShareToEncShareOwned};

/// Encryption to additive shares: every party keeps a random mask `M_i` as its
/// share and publishes the inner product of the ciphertext's mask with its
/// secret, minus `M_i`, plus its flood. The public shares are aggregated with
/// core `glwe_add_assign`, and one party adds them and the ciphertext's body to
/// its mask, so that the shares sum to the plaintext plus the input noise and
/// the floods.
///
/// A share is a plaintext read as a signed integer in its top-`k` window, the
/// torus value `M * 2^-k`. A mask is a uniform integer of `log_bound` bits. The
/// masks hide the noisy plaintext, of magnitude `B` with the input noise and
/// the aggregate flood, in the finalizing party's share: over `n`
/// coefficients, a shift by it moves the masks by at most `n * B / 2^log_bound`
/// in statistical distance, so `log_bound >= log2(B) + log2(n) + lambda`. The
/// reconstruction must not wrap: `B + parties * 2^log_bound < 2^(k - 1)`
/// coefficientwise.
///
/// The public share is a rank-0 core `GLWE`. Its `flood`, drawn from
/// `source_smudge` at the share's precision, hides the party's secret and the
/// input noise once the shares are reconstructed; size it as the smudging
/// section of `docs/mhe-contracts.md` describes. Share generation reads the
/// ciphertext's mask alone, which must be canonical and the session's honest
/// ciphertext's, as for
/// [`GLWEPrivateKeyswitchMHEProtocol`](crate::api::GLWEPrivateKeyswitchMHEProtocol).
pub trait GLWEEncToShareMHEProtocol<BE: Backend> {
    /// `ct_infos` is the ciphertext layout.
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Draws this party's mask into `secret` from `source_xm` and writes into
    /// `public` the inner product of `mask` with `sk`, minus the mask, plus
    /// fresh noise drawn with `flood`. The flood is added only to `public`, so
    /// it survives the cancellation of the masks. `secret` and `public` have the
    /// precision of `mask`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
        &self,
        public: &mut GLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        log_bound: usize,
        flood: SmudgingNoise,
        source_xm: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_enc_to_share_share_aggregate(&self, res: &mut GLWEEncToShareShareOwned<BE>, a: &GLWEEncToShareShareOwned<BE>);

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(&self) -> usize;

    /// Adds the body of `ct` and the aggregated public shares to `secret`, the
    /// finalizing party's share. The other parties keep their masks. Wipes the
    /// scratch used to normalize the private share before returning.
    fn mhe_glwe_enc_to_share_share_finalize<P, C>(
        &self,
        secret: &mut P,
        ct: &C,
        public: &GLWEEncToShareShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Additive shares to encryption: every party publishes the encryption of its
/// share under the common seed, and any party aggregates the shares and
/// finalizes the encryption of their sum under the ideal secret.
///
/// A share is a [`GLWEPatCompressed`](crate::layouts::GLWEPatCompressed), the
/// seeded encryption of the party's additive share raised to the output's
/// precision: the same integer in the top-`k` window, by `glwe_copy` into the
/// wider layout then `glwe_rsh` by the precision difference.
pub trait GLWEShareToEncMHEProtocol<BE: Backend> {
    /// `res_infos` is the output layout, `secret_infos` the share layout.
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(&self, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    /// Writes into `res` the encryption under `sk`, with its uniform components
    /// drawn from `seed`, of `secret` raised to the precision of `res`: the
    /// same integer in its top-`k` window. `seed` must be fresh for every
    /// conversion: two encryptions under one secret and one seed reveal the
    /// difference of their plaintexts.
    ///
    /// The error uses the `sigma` and `bound` from `enc_infos`, sampled at
    /// `res.k()` regardless of the precision in `enc_infos`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_share_to_enc_share_gen<P, S, E>(
        &self,
        res: &mut GLWEShareToEncShareOwned<BE>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout and seed.
    fn mhe_glwe_share_to_enc_share_aggregate(&self, res: &mut GLWEShareToEncShareOwned<BE>, a: &GLWEShareToEncShareOwned<BE>);

    fn mhe_glwe_share_to_enc_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into `res`, the canonical encryption of
    /// the sum of the parties' shares under the ideal secret.
    fn mhe_glwe_share_to_enc_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEShareToEncShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}
