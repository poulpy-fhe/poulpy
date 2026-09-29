use poulpy_core::{
    EncryptionInfos, SmudgingNoise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWERefreshShareOwned;

/// Collective refresh in one round: every party draws private integers `M_i`
/// and generates a share of two parts, the encryption-to-shares part, the inner
/// product of the ciphertext's mask with its secret, minus `M_i`, plus its
/// flood, and the shares-to-encryption part, the seeded encryption of `M_i` at
/// the output precision. Any party aggregates the shares and finalizes a fresh
/// encryption under the ideal secret of the same integer plaintext at the
/// output precision, plus the input noise, the floods and fresh encryption
/// noise.
///
/// The ciphertext's plaintext is read as a signed integer in its top-`k`
/// window, the torus value `m * 2^-k`, and the raise to the output precision
/// keeps the integer, so the reconstruction must not wrap, unlike
/// [`GLWEEncToShareMHEProtocol`](crate::api::GLWEEncToShareMHEProtocol)'s torus
/// shares. `M_i` is uniform on `log_bound` bits. It hides the noisy plaintext,
/// of magnitude `B` with the input noise and the aggregate flood: over `n`
/// coefficients, a shift by it moves `M_i` by at most `n * B / 2^log_bound` in
/// statistical distance, so `log_bound >= log2(B) + log2(n) + lambda`. No wrap
/// needs `B + parties * 2^log_bound < 2^(k - 1)` coefficientwise.
///
/// The `flood`, drawn from `source_smudge` at the ciphertext's precision, is
/// added to the encryption-to-shares part alone, so it survives the
/// cancellation of the `M_i` and the raise; size it as the smudging section of
/// `docs/mhe-contracts.md` describes. Share generation reads the ciphertext's
/// mask alone, which must be canonical and the session's honest ciphertext's,
/// as for [`GLWEPrivateKeyswitchMHEProtocol`](crate::api::GLWEPrivateKeyswitchMHEProtocol).
/// The seed of the shares-to-encryption part follows
/// [`GLWEShareToEncMHEProtocol`](crate::api::GLWEShareToEncMHEProtocol).
pub trait GLWERefreshMHEProtocol<BE: Backend> {
    /// `ct_infos` is the input ciphertext layout, `res_infos` the output layout.
    fn mhe_glwe_refresh_share_gen_tmp_bytes<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    /// Draws this party's `M_i` from `source_xm` and writes into `res` its
    /// share of the ciphertext of mask `mask`: the encryption-to-shares part
    /// at the precision of `mask`, flooded from `source_smudge`, and the
    /// shares-to-encryption part, the encryption under `sk` of `M_i` raised to
    /// the output precision, with its uniform components drawn from `seed` and
    /// its error drawn with `enc_infos` from `source_xe` at the output
    /// precision. `M_i` never leaves the call.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_refresh_share_gen<C, S, E>(
        &self,
        res: &mut GLWERefreshShareOwned<BE>,
        mask: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: SmudgingNoise,
        enc_infos: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layouts and seed.
    fn mhe_glwe_refresh_share_aggregate(&self, res: &mut GLWERefreshShareOwned<BE>, a: &GLWERefreshShareOwned<BE>);

    /// `res_infos` is the output layout.
    fn mhe_glwe_refresh_share_finalize_tmp_bytes<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res` the finalized shares-to-encryption parts with the
    /// body of `ct` plus the aggregated encryption-to-shares parts, raised to
    /// the precision of `res`, added to its body. `ct`, the
    /// encryption-to-shares part and `res` share their degree and radix, and
    /// the encryption-to-shares part has the precision of `ct`.
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
