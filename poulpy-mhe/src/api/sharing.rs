use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEEncToShareShareOwned, GLWEShareToEncShareOwned};

/// Encryption to additive shares: every party keeps a random mask as its share
/// and publishes its partial decryption minus the mask; the public shares are
/// aggregated with core `glwe_add_assign`, and one party finalizes its share
/// with them, so that the parties' shares sum to the plaintext plus input
/// noise and the parties' independent smudging noise.
///
/// A share is a plaintext read as a signed integer in its top-`k` window. A
/// mask is a uniform integer of `log_bound` bits: `log_bound` must exceed the
/// bit size of the noisy plaintext by the statistical hiding margin. The
/// plaintext, input noise, aggregate flood and masks must fit below `2^(k - 1)`.
///
/// `flood` follows the smudging contract of [`GLWEKeyswitchShare`](crate::api::GLWEKeyswitchMHEProtocol):
/// each party provisions Gaussian or uniform noise for the full statistical
/// margin, on a lattice fine enough to hide the input error. Bounded masks
/// alone do not protect the reconstructed decryption phase.
/// `source_xm` and `source_xe` must be independent private streams, never replayed.
pub trait GLWEEncToShareMHEProtocol<BE: Backend> {
    /// `ct_infos` is the ciphertext layout.
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Draws this party's mask into `secret` and writes into `public`, a rank-0
    /// GLWE, the inner product of the uniform components of `ct` with `sk`,
    /// minus the mask, plus fresh noise drawn with `flood`. The flood is added
    /// only to `public`; it survives mask cancellation. `secret` and `public`
    /// have the precision of `ct`.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_enc_to_share_share_gen<P, C, S, E>(
        &self,
        public: &mut GLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        ct: &C,
        sk: &S,
        log_bound: usize,
        flood: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_enc_to_share_share_aggregate(&self, res: &mut GLWEEncToShareShareOwned<BE>, a: &GLWEEncToShareShareOwned<BE>);

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(&self) -> usize;

    /// Adds the body of `ct` and the aggregated public shares to `secret`, the
    /// finalizing party's share. The other parties keep their masks.
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
