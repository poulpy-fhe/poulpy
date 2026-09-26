use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWEPreparedToBackendRef, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::layouts::GGLWEPatOwned;

/// Collective tensor (relinearization) key: every party publishes a share
/// built under the collective public key, the shares are aggregated with
/// [`PatAggregate`](crate::api::PatAggregate), and
/// [`PatFinalize`](crate::api::PatFinalize) expands the aggregate into the
/// tensor key of the ideal secret, the sum of the parties' secrets.
pub trait GLWETensorKeyShare<BE: Backend> {
    /// `res_infos` is the tensor key layout, `pk_infos` the public key layout.
    fn glwe_tensor_key_share_tmp_bytes<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    /// Writes this party's share into `res`, laid out as the tensor key: every
    /// entry is an encryption of zero under `pk` with a component of `sk` added
    /// to its masks, so that the aggregate encrypts the pairwise products of
    /// the ideal secret's components.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_key_share<S, K, E>(
        &self,
        res: &mut GGLWEPatOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;
}
