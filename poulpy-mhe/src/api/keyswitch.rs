use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Collective key switching to a secret key: every party publishes a share,
/// the shares are aggregated with core `glwe_add_assign`, and any party
/// finalizes the ciphertext under the ideal output secret, the sum of the
/// parties' output secrets.
pub trait GLWEKeyswitchShare<BE: Backend> {
    /// `infos` is the ciphertext layout; the share has that layout at rank 0.
    fn glwe_keyswitch_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes this party's share into `res`, a rank-0 GLWE: the inner product
    /// of the masks of `ct` with `sk_in - sk_out`, plus smudging noise drawn
    /// with `flood`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_keyswitch_share<R, C, S1, S2, E>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_keyswitch_finalize_tmp_bytes(&self) -> usize;

    /// Writes into `res` the ciphertext `ct` with the aggregated shares added
    /// to its body.
    fn glwe_keyswitch_finalize<R, C, H>(&self, res: &mut R, ct: &C, share: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos;
}
