use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::api::CKKSSubOps;

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSSubImpl};

impl<BE: Backend + CKKSSubImpl> CKKSSubOps<BE> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_tmp_bytes_impl(self, res_size)
    }

    fn ckks_sub_pt_vec_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_pt_vec_tmp_bytes_impl(self, res_size)
    }

    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_into_impl(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_assign_impl(self, dst, a, scratch)
    }

    fn ckks_sub_one_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_one_tmp_bytes_impl(self, res_size)
    }

    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_sub_one_assign_impl(self, dst, scratch)
    }

    fn ckks_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_into_impl(self, dst, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_assign_impl(self, dst, pt, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_sub_pt_const_tmp_bytes_impl(self, res_size)
    }

    fn ckks_sub_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_const_into_impl(self, dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_sub_pt_const_assign<Dst, P>(
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
        BE::ckks_sub_pt_const_assign_impl(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}
