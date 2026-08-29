use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey},
};
use poulpy_hal::layouts::{Backend, GaloisElement, Module, ScratchArena};

use crate::{CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSRotateImpl};

use crate::api::CKKSRotateOps;

impl<BE: Backend + CKKSRotateImpl<BE>> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE> + GaloisElement,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes_impl(self, ct_infos, key_infos)
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
        let p = self.galois_element(k);
        let key = keys
            .get_automorphism_key(p, src.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
                k: src.k().into(),
            })?;
        BE::ckks_rotate_into_impl(self, dst, src, &&key, scratch)
    }

    fn ckks_rotate_assign<Dst, H>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let p = self.galois_element(k);
        let key = keys
            .get_automorphism_key(p, dst.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
                k: dst.k().into(),
            })?;
        BE::ckks_rotate_assign_impl(self, dst, &&key, scratch)
    }
}
