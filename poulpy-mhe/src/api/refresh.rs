use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWERefreshShareOwned;

/// Collective refresh in one round: every party generates a share, an
/// encryption-to-shares part and a shares-to-encryption part built from one
/// mask, and any party aggregates the shares and finalizes a fresh encryption
/// of the plaintext under the ideal secret: the same integer in the top-`k`
/// window of the output precision, plus the input, smudging and fresh
/// encryption noise.
///
/// The mask bound and `flood` follow [`GLWEEncToShareMHEProtocol`](crate::api::GLWEEncToShareMHEProtocol).
/// Flooding is applied in the input frame and survives the integer-preserving
/// raise. `enc_infos` controls the independent S2E noise in the output frame.
pub trait GLWERefreshMHEProtocol<BE: Backend> {
    /// `ct_infos` is the input ciphertext layout, `res_infos` the output layout.
    fn mhe_glwe_refresh_share_gen_tmp_bytes<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    /// Draws this party's mask and writes into `res` its public share of `ct`,
    /// the E2S part, a rank-0 GLWE at the precision of `ct`, and the encryption
    /// of the mask, the S2E part, at the output layout, with its uniform components drawn
    /// from a fresh `seed` shared by this invocation's parties. Adds fresh
    /// `flood` noise only to the E2S part, never to the mask encrypted by the S2E part.
    /// The mask never leaves the call. Both error draws consume the advancing
    /// private `source_xe`, independently of the private mask stream `source_xm`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_refresh_share_gen<C, S, E1, E2>(
        &self,
        res: &mut GLWERefreshShareOwned<BE>,
        ct: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: &E1,
        enc_infos: &E2,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    /// `res_infos` is the output layout.
    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layouts and S2E seed.
    fn mhe_glwe_refresh_share_aggregate(&self, res: &mut GLWERefreshShareOwned<BE>, a: &GLWERefreshShareOwned<BE>);

    fn mhe_glwe_refresh_share_finalize_tmp_bytes<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res` the finalized S2E parts with the body of `ct` plus the
    /// aggregated E2S parts, raised to the precision of `res`, added to its
    /// body. `ct`, the E2S part and `res` share their degree and radix, and the
    /// E2S part has the precision of `ct`.
    fn mhe_glwe_refresh_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWERefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}
