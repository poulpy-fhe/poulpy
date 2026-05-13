use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::UnnormalizedCKKSCiphertext;
use crate::leveled::api::{CKKSAddOps, CKKSAddOpsUnnormalized};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, oep::CKKSAddImpl};

impl<BE: Backend + CKKSAddImpl<BE>> CKKSAddOps<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self) -> usize {
        BE::ckks_add_tmp_bytes(self)
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

    fn ckks_add_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_add_one_assign(self, dst, scratch)
    }

    fn ckks_add_pt_vec_tmp_bytes(&self) -> usize {
        BE::ckks_add_pt_vec_tmp_bytes(self)
    }

    fn ckks_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_vec_into(self, dst, a, pt, scratch)
    }

    fn ckks_add_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_vec_assign(self, dst, pt, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize {
        BE::ckks_add_pt_const_tmp_bytes(self)
    }

    fn ckks_add_pt_const_into<Dst, A, P>(
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_const_into(self, dst, a, dst_coeff, pt, pt_coeff, scratch)
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_const_assign(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}

impl<BE: Backend + CKKSAddImpl<BE>> CKKSAddOpsUnnormalized<BE> for Module<BE> {
    fn ckks_add_into_unnormalized<Dst, A, B>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_into_unnormalized(self, dst, a, b, scratch)
    }

    fn ckks_add_assign_unnormalized<Dst, A>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        BE::ckks_add_assign_unnormalized(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_vec_into_unnormalized(self, dst, a, pt, scratch)
    }

    fn ckks_add_pt_vec_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_vec_assign_unnormalized(self, dst, pt, scratch)
    }

    fn ckks_add_pt_const_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_const_into_unnormalized(self, dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_add_pt_const_assign_unnormalized(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}
