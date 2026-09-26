use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWEPreparedToBackendRef, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWETensorKeyShare, layouts::GGLWEPatOwned, oep::GLWETensorKeyShareImpl};

impl<BE: Backend + GLWETensorKeyShareImpl> GLWETensorKeyShare<BE> for Module<BE> {
    fn glwe_tensor_key_share_tmp_bytes<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_tensor_key_share_tmp_bytes(self, res_infos, pk_infos)
    }

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
        E: EncryptionInfos,
    {
        BE::glwe_tensor_key_share(self, res, sk, pk, enc_infos, source_xu, source_xe, scratch)
    }
}
