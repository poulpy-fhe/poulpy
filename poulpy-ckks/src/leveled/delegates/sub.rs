use anyhow::Result;
use poulpy_core::{GLWEShift, GLWESub, ScratchTakeCore, layouts::GLWEInfos};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    oep::CKKSImpl,
};

use crate::leveled::{api::CKKSSubOps, oep::CKKSSubOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSSubOps<BE> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSSubOep::ckks_sub_tmp_bytes(self)
    }

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSSubOep::ckks_sub_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_sub_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_into(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_assign(self, dst, a, scratch)
    }

    fn ckks_sub_pt_vec_znx_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_znx_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSSubOep::ckks_sub_pt_vec_rnx_tmp_bytes(self, res, a, b)
    }

    fn ckks_sub_pt_vec_rnx_into<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSSubOep::ckks_sub_pt_vec_rnx_into(self, dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_vec_rnx_assign<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSSubOep::ckks_sub_pt_vec_rnx_assign(self, dst, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSSubOep::ckks_sub_pt_const_tmp_bytes(self)
    }

    fn ckks_sub_pt_const_znx_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_pt_const_znx_into(self, dst, a, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_znx_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSSubOep::ckks_sub_pt_const_znx_assign(self, dst, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_rnx_into<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSSubOep::ckks_sub_pt_const_rnx_into(self, dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_sub_pt_const_rnx_assign<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSSubOep::ckks_sub_pt_const_rnx_assign(self, dst, cst_rnx, prec, scratch)
    }
}

use crate::leveled::api::CKKSSubOpsUnsafe;
use crate::leveled::default::CKKSSubDefault;

unsafe impl<BE: Backend> CKKSSubOpsUnsafe<BE> for Module<BE>
where
    Module<BE>: CKKSSubDefault<BE>,
{
    unsafe fn ckks_sub_into_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_into_unsafe_default(dst, a, b, scratch)
    }

    unsafe fn ckks_sub_assign_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_assign_unsafe_default(dst, a, scratch)
    }

    unsafe fn ckks_sub_pt_vec_znx_into_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)
    }

    unsafe fn ckks_sub_pt_vec_znx_assign_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)
    }

    unsafe fn ckks_sub_pt_vec_rnx_into_unsafe<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_into_unsafe_default(dst, a, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_sub_pt_vec_rnx_assign_unsafe<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_assign_unsafe_default(dst, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_sub_pt_const_znx_into_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_const_znx_into_unsafe_default(dst, a, cst_znx, scratch)
    }

    unsafe fn ckks_sub_pt_const_znx_assign_unsafe(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)
    }

    unsafe fn ckks_sub_pt_const_rnx_into_unsafe<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_pt_const_rnx_into_unsafe_default(dst, a, cst_rnx, prec, scratch)
    }

    unsafe fn ckks_sub_pt_const_rnx_assign_unsafe<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_pt_const_rnx_assign_unsafe_default(dst, cst_rnx, prec, scratch)
    }
}
