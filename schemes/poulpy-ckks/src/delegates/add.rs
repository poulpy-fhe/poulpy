use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::api::CKKSAddOps;

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSAddImpl};

impl<BE: Backend + CKKSAddImpl> CKKSAddOps<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_add_tmp_bytes(self, res_size)
    }

    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_into(self, dst, a, b, scratch)
    }

    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_assign(self, dst, a, scratch)
    }

    fn ckks_add_one_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_add_one_tmp_bytes(self, res_size)
    }

    fn ckks_add_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_add_one_assign(self, dst, scratch)
    }

    fn ckks_add_pt_vec_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_add_pt_vec_tmp_bytes(self, res_size)
    }

    fn ckks_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_add_pt_vec_into(self, dst, a, pt, scratch)
    }

    fn ckks_add_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_add_pt_vec_assign(self, dst, pt, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_add_pt_const_tmp_bytes(self, res_size)
    }

    fn ckks_add_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_add_pt_const_into(self, dst, dst_coeff, a, pt, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_add_pt_const_assign(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}
