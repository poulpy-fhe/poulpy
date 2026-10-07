use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GLWETensorKeyShareOwned;

/// Collective tensor (relinearization) key: every party generates a share
/// under the collective public key, and any party aggregates the shares and
/// finalizes the tensor key of the ideal secret, the sum of the parties'
/// secrets.
///
/// A share is a `GGLWEPat` laid out as core's `GLWETensorKey`. Its entries are
/// public-key encryptions, unseeded, so no mask is common across parties.
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
    ///
    /// `pk` must be the finalized collective public key of the session: under
    /// another key, its holder decrypts the share and reads `sk`'s components.
    /// It must also be at least as precise as the tensor key; build it at the
    /// key's precision. `source_xu` (ephemerals) and `source_xe` (errors) must
    /// be private, independently seeded for each party and purpose; never
    /// replay their streams.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_tensor_key_share_gen<S, K>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

    /// Adds share `a` into `res`, which starts as the first share. The shares
    /// must have the same layout.
    fn mhe_glwe_tensor_key_share_aggregate(&self, res: &mut GLWETensorKeyShareOwned<BE>, a: &GLWETensorKeyShareOwned<BE>);

    fn mhe_glwe_tensor_key_share_finalize_tmp_bytes(&self) -> usize;

    /// Expands the aggregated shares into the canonical tensor key `res`, which
    /// must have the share's layout. The masks are sums of the parties'
    /// encryptions, so every column is normalized.
    fn mhe_glwe_tensor_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWETensorKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}
