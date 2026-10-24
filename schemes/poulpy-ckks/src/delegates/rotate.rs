use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSRotateOps, oep::CKKSRotateImpl};

impl<BE: Backend + CKKSRotateImpl> CKKSRotateOps<BE> for Module<BE> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into<Dst, Src, H>(
        &self,
        k: i64,
        dst: &mut Dst,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_rotate_into(self, k, dst, src, keys, scratch)
    }

    fn ckks_rotate_assign<Dst, H>(&self, k: i64, dst: &mut Dst, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_rotate_assign(self, k, dst, keys, scratch)
    }
}
