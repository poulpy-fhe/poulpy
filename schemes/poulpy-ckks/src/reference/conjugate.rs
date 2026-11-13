use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSInfos, SetCKKSInfos,
    api::CKKSRoundOps,
    reference::rotate::{automorphism_assign, automorphism_into, automorphism_tmp_bytes},
};

pub trait CKKSConjugateReference<BE: Backend> {
    fn ckks_conjugate_with_key_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: crate::CKKSCtBounds,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
    {
        automorphism_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_with_key_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        automorphism_into("ckks_conjugate_into", self, dst, src, key, scratch)
    }

    fn ckks_conjugate_with_key_assign_reference<Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    {
        automorphism_assign("ckks_conjugate_assign", self, dst, key, scratch)
    }
}

impl<BE: Backend> CKKSConjugateReference<BE> for poulpy_hal::layouts::Module<BE> {}
