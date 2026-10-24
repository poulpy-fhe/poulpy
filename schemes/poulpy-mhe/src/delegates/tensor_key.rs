use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWETensorKeyMHEProtocol, layouts::GLWETensorKeyShareOwned, oep::GLWETensorKeyMHEProtocolImpl};

impl<BE: Backend + GLWETensorKeyMHEProtocolImpl> GLWETensorKeyMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos,
    {
        BE::mhe_glwe_tensor_key_share_gen_tmp_bytes(self, res_infos, pk_infos)
    }

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
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_tensor_key_share_gen(self, res, sk, pk, source_xu, source_xe, scratch)
    }

    fn mhe_glwe_tensor_key_share_aggregate(&self, res: &mut GLWETensorKeyShareOwned<BE>, a: &GLWETensorKeyShareOwned<BE>) {
        BE::mhe_glwe_tensor_key_share_aggregate(self, res, a)
    }

    fn mhe_glwe_tensor_key_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_tensor_key_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_tensor_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWETensorKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::mhe_glwe_tensor_key_share_finalize(self, res, share, scratch)
    }
}
