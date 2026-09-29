use poulpy_core::{
    EncryptionInfos,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWETensorKeyShareOwned;

/// Collective tensor (relinearization) key: every party generates a share
/// under the collective public key, and any party aggregates the shares and
/// finalizes the tensor key of the ideal secret, the sum of the parties'
/// secrets.
pub trait GLWETensorKeyMHEProtocol<BE: Backend> {
    /// `res_infos` is the tensor key layout, `pk_infos` the public key layout.
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    /// Writes this party's share into `res`: every entry is an encryption of
    /// zero under `pk` with a component of `sk` added to its masks, so that the
    /// aggregate encrypts the pairwise products of the ideal secret's
    /// components.
    /// `pk` must be at least as precise as the tensor key, build the collective
    /// public key at the key's precision.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_tensor_key_share_gen<S, K, E>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_tensor_key_share_aggregate(&self, res: &mut GLWETensorKeyShareOwned<BE>, a: &GLWETensorKeyShareOwned<BE>);

    fn mhe_glwe_tensor_key_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical tensor key `res`, which
    /// must have the share's layout.
    fn mhe_glwe_tensor_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWETensorKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}
