use anyhow::Result;
use poulpy_core::{
    GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena},
    oep::HalVecZnxImpl,
};

use crate::layouts::{CKKSCiphertext, UnnormalizedCKKSCiphertext};
use crate::leveled::{
    api::{CKKSSubOps, CKKSSubOpsUnnormalized},
    default::{CKKSPlaintextDefault, CKKSSubDefault},
};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, oep::CKKSSubImpl};

impl<BE: Backend + CKKSSubImpl<BE>> CKKSSubOps<BE> for Module<BE>
where
    Module<BE>: GLWEShift<BE>
        + GLWESub<BE>
        + GLWENormalize<BE>
        + CKKSPlaintextDefault<BE>
        + ModuleN
        + VecZnxRshSubBackend<BE>
        + VecZnxRshSubCoeffIntoBackend<BE>
        + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_sub_tmp_bytes(&self) -> usize {
        BE::ckks_sub_tmp_bytes(self)
    }

    fn ckks_sub_pt_vec_tmp_bytes(&self) -> usize {
        BE::ckks_sub_pt_vec_tmp_bytes(self)
    }

    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_into(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_assign(self, dst, a, scratch)
    }

    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_sub_one_assign(self, dst, scratch)
    }

    fn ckks_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_pt_vec_into(self, dst, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_pt_vec_assign(self, dst, pt, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize {
        BE::ckks_sub_pt_const_tmp_bytes(self)
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_pt_const_into(self, dst, a, dst_coeff, pt, pt_coeff, scratch)
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_sub_pt_const_assign(self, dst, dst_coeff, pt, pt_coeff, scratch)
    }
}

impl<BE: Backend + HalVecZnxImpl<BE>> CKKSSubOpsUnnormalized<BE> for Module<BE>
where
    Module<BE>: CKKSSubDefault<BE>
        + GLWEShift<BE>
        + GLWESub<BE>
        + CKKSPlaintextDefault<BE>
        + ModuleN
        + VecZnxRshSubBackend<BE>
        + VecZnxRshSubCoeffIntoBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_sub_into_unnormalized<Dst, A, B>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_sub_into_unsafe_default(&mut dst.inner, a, b, scratch)
    }

    fn ckks_sub_assign_unnormalized<Dst, A>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        self.ckks_sub_assign_unsafe_default(&mut dst.inner, a, scratch)
    }

    fn ckks_sub_pt_vec_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_sub_pt_vec_into_unsafe_default(&mut dst.inner, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_sub_pt_vec_assign_unsafe_default(&mut dst.inner, pt, scratch)
    }

    fn ckks_sub_pt_const_into_unnormalized<Dst, A, P>(
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
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_sub_pt_const_into_unsafe_default(&mut dst.inner, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_sub_pt_const_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_sub_pt_const_assign_unsafe_default(&mut dst.inner, dst_coeff, pt, pt_coeff, scratch)
    }
}
