use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::ckks::layouts::CKKSRefreshShareOwned;

/// Collective CKKS refresh in one round: every party draws private integers
/// `M_i` and generates a share of two parts, the encryption-to-shares part, the
/// inner product of the ciphertext's mask with its secret, minus `M_i`, plus
/// fresh Gaussian noise of standard deviation 3.2, and the shares-to-encryption
/// part, the seeded encryption of `M_i` at the output precision. Any party
/// aggregates the shares and finalizes a fresh encryption under the ideal secret
/// of the same integer plaintext at the output precision, plus the input noise
/// and both parts' fresh noise.
///
/// CKKS reads the ciphertext's plaintext as a signed integer in its top-`k`
/// window, the torus value `m * 2^-k`, and its modulus raise to the output
/// precision keeps the integer, so the reconstruction must not wrap, unlike
/// [`GLWEEncToShareMHEProtocol`](crate::api::GLWEEncToShareMHEProtocol)'s torus
/// shares. `M_i` is uniform on `log_bound` bits. It hides the noisy plaintext,
/// of magnitude `B` with the input noise and the aggregate encryption-to-shares
/// noise: over `n` coefficients, a shift by it moves `M_i` by at most
/// `n * B / 2^log_bound` in statistical distance, so
/// `log_bound >= log2(B) + log2(n) + lambda`. No wrap
/// needs `B + parties * 2^log_bound < 2^(k - 1)` coefficientwise. The caller
/// sizes `log_bound`; neither bound is checked, as `B` and the party count are
/// unknown here. Repeated refreshes require a statistical budget over all calls.
///
/// The private masks statistically hide the opened noisy plaintext. With all
/// but one party corrupt, the opening, the output and the corrupt parties'
/// contributions determine the honest party's two public shares. No smudging
/// flood is needed for privacy beyond the input and output ciphertexts in the
/// passive model. The output retains the input error; releasing its approximate
/// decryption requires separate protection as described in `docs/mhe-contracts.md`.
///
/// Both parts retain independent ordinary noise: their masks cancel when the
/// public shares are combined. The encryption-to-shares noise uses sigma 3.2
/// and bound `6 * 3.2` at the ciphertext's precision. It is added only to the
/// public part, so it survives mask cancellation and the integer-preserving
/// raise. Successive fresh seeds from `source_xe` supply this noise and the
/// shares-to-encryption noise configured by `enc_infos` at the output precision.
/// Share generation reads the ciphertext's mask alone, which must be canonical
/// and the session's honest ciphertext's, as for
/// [`GLWEPrivateKeyswitchMHEProtocol`](crate::api::GLWEPrivateKeyswitchMHEProtocol).
/// The seed of the shares-to-encryption part follows
/// [`GLWEShareToEncMHEProtocol`](crate::api::GLWEShareToEncMHEProtocol).
pub trait CKKSRefreshMHEProtocol<BE: Backend> {
    /// `ct_infos` is the input ciphertext layout, `res_infos` the output layout.
    fn mhe_ckks_refresh_share_gen_tmp_bytes<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    /// Draws this party's `M_i` from `source_xm` and writes into `res` its
    /// share of the ciphertext of mask `mask`: the encryption-to-shares part
    /// at the precision of `mask`, with automatic sigma-3.2 Gaussian noise, and the
    /// shares-to-encryption part, the encryption under `sk` of `M_i` raised to
    /// the output precision, with its uniform components drawn from `seed` and
    /// its error drawn with `enc_infos` from `source_xe` at the output
    /// precision. The private `source_xe` supplies independent noise for both
    /// parts and must be independent of `source_xm`. `M_i` never leaves the call.
    #[allow(clippy::too_many_arguments)]
    fn mhe_ckks_refresh_share_gen<C, S, E>(
        &self,
        res: &mut CKKSRefreshShareOwned<BE>,
        mask: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        enc_infos: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layouts and seed.
    fn mhe_ckks_refresh_share_aggregate(&self, res: &mut CKKSRefreshShareOwned<BE>, a: &CKKSRefreshShareOwned<BE>);

    /// `res_infos` is the output layout.
    fn mhe_ckks_refresh_share_finalize_tmp_bytes<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res` the finalized shares-to-encryption parts with the
    /// body of `ct` plus the aggregated encryption-to-shares parts, raised to
    /// the precision of `res`, added to its body. `ct`, the
    /// encryption-to-shares part and `res` share their degree and radix, and
    /// the encryption-to-shares part has the precision of `ct`.
    fn mhe_ckks_refresh_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &CKKSRefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}
