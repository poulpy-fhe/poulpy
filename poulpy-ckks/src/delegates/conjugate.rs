use crate::CKKSResult as Result;
use crate::{api::CKKSModuleInfos, ckks_ensure};
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSConjugateImpl, reference::paco::ops::conj_rotate_galois_element,
};

use crate::api::CKKSConjugateOps;

impl<BE: Backend + CKKSConjugateImpl> CKKSConjugateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
{
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
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_conjugate_rotate_into", keys.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_conjugate_rotate_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_conjugate_rotate_into", src)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "conjugation requires the standard CKKS ring"
        );
        let p: i64 = conj_rotate_galois_element(k, self.cyclotomic_order());
        let key = keys
            .get_automorphism_key(p, src.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "conjugate",
                rotation: k,
                k: src.k().into(),
            })?;
        BE::ckks_conjugate_into_impl(self, dst, src, &key, scratch)
    }

    fn ckks_conjugate_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_conjugate_assign", keys.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_conjugate_assign", dst)?;
        ckks_ensure!(
            !self.ckks_is_conjugate_invariant(),
            "conjugation requires the standard CKKS ring"
        );
        let key = keys
            .get_automorphism_key(-1, dst.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "conjugate_assign",
                rotation: 0,
                k: dst.k().into(),
            })?;
        BE::ckks_conjugate_assign_impl(self, dst, &key, scratch)
    }
}
