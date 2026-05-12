use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSRotateImpl};

use crate::api::CKKSRotateOps;

impl<BE: Backend + CKKSRotateImpl<BE>> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into<'s, Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
            })?;
        BE::ckks_rotate_into(self, dst, src, key, scratch)
    }

    fn ckks_rotate_assign<'s, Dst, H, K>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'s, BE>) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
            })?;
        BE::ckks_rotate_assign(self, dst, key, scratch)
    }
}
