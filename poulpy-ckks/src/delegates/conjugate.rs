use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSConjugateImpl};

use crate::api::CKKSConjugateOps;

impl<BE: Backend + CKKSConjugateImpl<BE>> CKKSConjugateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_conjugate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_into<'s, Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        BE::ckks_conjugate_into(self, dst, src, key, scratch)
    }

    fn ckks_conjugate_assign<'s, Dst, K>(&self, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'s, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        BE::ckks_conjugate_assign(self, dst, key, scratch)
    }
}
