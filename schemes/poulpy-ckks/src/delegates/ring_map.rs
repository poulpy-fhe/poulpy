use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCIRingMapOps, layouts::CKKSCiphertext, oep::CKKSCIRingMapImpl};

impl<BE: Backend + CKKSCIRingMapImpl> CKKSCIRingMapOps<BE> for Module<BE> {
    fn ckks_ci_embed_tmp_bytes<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::ckks_ci_embed_tmp_bytes(self, res_infos, key_infos)
    }

    fn ckks_ci_embed<Dst, D, K>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        key: &GLWECIEmbedKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
        K: Data,
        GLWECIEmbedKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        BE::ckks_ci_embed(self, dst, src, key, scratch)
    }

    fn ckks_ci_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::ckks_ci_trace_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn ckks_ci_trace<D, Src, K>(
        &self,
        dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        src: &Src,
        key: &GLWECITraceKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: Data,
        GLWECITraceKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        BE::ckks_ci_trace(self, dst, src, key, scratch)
    }
}
