use crate::CKKSResult as Result;
use poulpy_core::layouts::GetAutomorphismKey;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSConjugateImpl};

use crate::api::CKKSConjugateOps;

impl<BE: Backend + CKKSConjugateImpl> CKKSConjugateOps<BE> for Module<BE> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_conjugate_tmp_bytes_impl(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_rotate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_conjugate_rotate_into_impl(self, dst, src, k, keys, scratch)
    }

    fn ckks_conjugate_assign<Dst, H>(&self, dst: &mut Dst, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_conjugate_with_keys_assign_impl(self, dst, keys, scratch)
    }
}
