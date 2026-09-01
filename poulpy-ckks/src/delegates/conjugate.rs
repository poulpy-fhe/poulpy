use crate::CKKSResult as Result;
use poulpy_core::layouts::GetAutomorphismKey;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, Normalized, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, default::paco::ops::conj_rotate_galois_element, oep::CKKSConjugateImpl,
};

use crate::api::CKKSConjugateOps;

impl<BE: Backend + CKKSConjugateImpl<BE>> CKKSConjugateOps<BE> for Module<BE>
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
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
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

    fn ckks_conjugate_assign<Dst, H>(&self, dst: &mut Dst, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
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
