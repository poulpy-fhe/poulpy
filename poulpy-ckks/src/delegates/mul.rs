use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use crate::api::CKKSMulOps;

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSPreparedLeft, CKKSPreparedRight},
    oep::CKKSMulImpl,
};

impl<BE: Backend + CKKSMulImpl> CKKSMulOps<BE> for Module<BE> {
    fn ckks_mul_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // Sizing takes the key, not a decomposition: which effective `dsize` the
        // helper dispatches at is resolved during execution, and the budget
        // below covers every one this key admits.
        BE::ckks_mul_tmp_bytes(self, res, a, b, tsk)
    }

    fn ckks_square_tmp_bytes<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_square_tmp_bytes(self, res, a, tsk)
    }

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_vec_tmp_bytes(self, res, a, b.k())
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_const_tmp_bytes(self, res, a, b.k())
    }

    fn ckks_mul_into<Dst, A, B, H>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        BE::ckks_mul_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        BE::ckks_mul_assign(self, dst, a, tsk, scratch)
    }

    fn ckks_prepare_tmp_bytes<A>(&self, a: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_prepare_tmp_bytes(self, a)
    }

    fn ckks_prepared_left_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        BE::ckks_prepared_left_bytes_of_from_infos(self, infos, hint)
    }

    fn ckks_prepared_right_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        BE::ckks_prepared_right_bytes_of_from_infos(self, infos, hint)
    }

    fn ckks_prepared_left_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedLeft<BE>
    where
        A: GLWEInfos,
    {
        BE::ckks_prepared_left_alloc_from_infos(self, infos, hint)
    }

    fn ckks_prepared_right_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedRight<BE>
    where
        A: GLWEInfos,
    {
        BE::ckks_prepared_right_alloc_from_infos(self, infos, hint)
    }

    fn ckks_prepare_left_into<A>(&self, res: &mut CKKSPreparedLeft<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_left_into(self, res, a, scratch)
    }

    fn ckks_prepare_right_into<A>(&self, res: &mut CKKSPreparedRight<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_right_into(self, res, a, scratch)
    }

    fn ckks_prepare_left<A>(&self, a: &A, hint: PrepareHint, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedLeft<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_left(self, a, hint, scratch)
    }

    fn ckks_prepare_right<A>(&self, a: &A, hint: PrepareHint, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_right(self, a, hint, scratch)
    }

    fn ckks_mul_prepared_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        BE::ckks_mul_prepared_assign(self, dst, prepared, tsk, scratch)
    }

    fn ckks_square_into<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        BE::ckks_square_into(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign<Dst, H>(&self, dst: &mut Dst, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        BE::ckks_square_assign(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_into(self, dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
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
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
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
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_assign(self, dst, pt, pt_coeff, scratch)
    }
}
