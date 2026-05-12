use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::api::CKKSMulOps;

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, oep::CKKSMulImpl};

impl<BE: Backend + CKKSMulImpl<BE>> CKKSMulOps<BE> for Module<BE>
where
    Module<BE>: GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxCopyBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_mul_tmp_bytes(self, res, tsk)
    }

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_square_tmp_bytes(self, res, tsk)
    }

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        let b = b.meta();
        BE::ckks_mul_pt_vec_tmp_bytes(self, res, a, &b)
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        let b = b.meta();
        BE::ckks_mul_pt_const_tmp_bytes(self, res, a, &b)
    }

    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_mul_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_mul_assign(self, dst, a, tsk, scratch)
    }

    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_square_into(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_square_assign(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_into(self, dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_assign(self, dst, pt, scratch)
    }

    fn ckks_mul_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_into(self, dst, a, pt, pt_coeff, scratch)
    }

    fn ckks_mul_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_assign(self, dst, pt, pt_coeff, scratch)
    }
}
