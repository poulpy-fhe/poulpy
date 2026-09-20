use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, GaloisElement, Module, ScratchArena};

use crate::{CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSRotateImpl};

use crate::api::{CKKSModuleInfos, CKKSRotateOps};

impl<BE: Backend + CKKSRotateImpl> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE> + GaloisElement,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes_impl(self, ct_infos, key_infos).max(self.glwe_shift_tmp_bytes(ct_infos.size()))
    }

    fn ckks_rotate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_rotate_into", keys.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_rotate_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_rotate_into", src)?;
        let p = self.ckks_galois_element(k);
        if p == 1 {
            return crate::ckks_shift_stamp_unary(self, "rotate", dst, src, 0, 0, 0, scratch);
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

    fn ckks_rotate_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        k: i64,
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_rotate_assign", keys.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_rotate_assign", dst)?;
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
