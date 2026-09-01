use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWE, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::Unnormalized;
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::api::CKKSSubOps;
use crate::layouts::UnnormalizedCKKSCiphertext;

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, oep::CKKSSubImpl};

impl<BE: Backend + CKKSSubImpl<BE>> CKKSSubOps<BE> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self) -> usize {
        BE::ckks_sub_tmp_bytes_impl(self)
    }

    fn ckks_sub_pt_vec_tmp_bytes(&self) -> usize {
        BE::ckks_sub_pt_vec_tmp_bytes_impl(self)
    }

    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    {
        BE::ckks_sub_into_impl(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    {
        BE::ckks_sub_assign_impl(self, dst, a, scratch)
    }

    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_sub_one_assign_impl(self, dst, scratch)
    }

    fn ckks_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_into_impl(self, dst, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_assign_impl(self, dst, pt, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize {
        BE::ckks_sub_pt_const_tmp_bytes_impl(self)
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_const_assign_impl(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
    fn ckks_sub_into_unnormalized<Dst, A, B>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    {
        BE::ckks_sub_into_unnormalized_impl(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign_unnormalized<Dst, A>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSInfos,
    {
        BE::ckks_sub_assign_unnormalized_impl(self, dst, a, scratch)
    }

    fn ckks_sub_pt_vec_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_into_unnormalized_impl(self, dst, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_vec_assign_unnormalized_impl(self, dst, pt, scratch)
    }

    fn ckks_sub_pt_const_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_const_into_unnormalized_impl(self, dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_sub_pt_const_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_sub_pt_const_assign_unnormalized_impl(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}
