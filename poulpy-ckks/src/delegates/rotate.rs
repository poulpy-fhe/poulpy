use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSRotateImpl};

use crate::api::CKKSCopyOps;
use crate::api::CKKSModuleInfos;
use crate::api::CKKSRotateOps;

impl<BE: Backend + CKKSRotateImpl> CKKSRotateOps<BE> for Module<BE> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes_impl(self, ct_infos, key_infos).max(BE::ckks_copy_tmp_bytes_impl(self, ct_infos, ct_infos))
    }

    fn ckks_rotate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        let p = self.ckks_galois_element(k);
        if p == 1 {
            return self.ckks_copy(dst, src, scratch);
        }
        let key = keys
            .get_automorphism_key(p, src.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
                k: src.k().into(),
            })?;
        BE::ckks_rotate_into_impl(self, dst, src, &key, scratch)
    }

    fn ckks_rotate_assign<Dst, H>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let p = self.ckks_galois_element(k);
        if p == 1 {
            return Ok(());
        }
        let key = keys
            .get_automorphism_key(p, dst.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
                k: dst.k().into(),
            })?;
        BE::ckks_rotate_assign_impl(self, dst, &key, scratch)
    }
}
