use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
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

use crate::leveled::{api::CKKSMulOps, oep::CKKSMulOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSMulOps<BE> for Module<BE> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        CKKSMulOep::ckks_mul_tmp_bytes(self, res, tsk)
    }

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        CKKSMulOep::ckks_square_tmp_bytes(self, res, tsk)
    }

    fn ckks_mul_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_znx_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_rnx_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        CKKSMulOep::ckks_mul_pt_const_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_assign(self, dst, a, tsk, scratch)
    }

    fn ckks_square_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_square_into(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_square_assign(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_znx_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_vec_znx_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_mul_pt_vec_rnx_into<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_rnx_into(self, dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_mul_pt_vec_rnx_assign<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_vec_rnx_assign(self, dst, pt_rnx, prec, scratch)
    }

    fn ckks_mul_pt_const_znx_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_const_znx_into(self, dst, a, cst_znx, scratch)
    }

    fn ckks_mul_pt_const_znx_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSMulOep::ckks_mul_pt_const_znx_assign(self, dst, cst_znx, scratch)
    }

    fn ckks_mul_pt_const_rnx_into<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSMulOep::ckks_mul_pt_const_rnx_into(self, dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_mul_pt_const_rnx_assign<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSMulOep::ckks_mul_pt_const_rnx_assign(self, dst, cst_rnx, prec, scratch)
    }
}
