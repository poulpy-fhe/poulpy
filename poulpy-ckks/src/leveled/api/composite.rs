use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddAssign, VecZnxRshAddInto},
    layouts::{Backend, DataMut, DataRef, Scratch},
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
    leveled::api::{
        add::{CKKSAddOps, CKKSAddOpsUnsafe},
        mul::CKKSMulOps,
        sub::CKKSSubOps,
    },
    oep::CKKSImpl,
};

pub trait CKKSAddManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>;

    fn ckks_add_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsUnsafe<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

pub trait CKKSMulManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>;

    fn ckks_mul_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

pub trait CKKSMulAddOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_add_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEAdd + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_mul_add_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_add_pt_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

pub trait CKKSMulSubOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_mul_sub_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_pt_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

pub trait CKKSDotProductOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_ct<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + VecZnxAddAssign
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_pt_vec_znx<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_pt_vec_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_dot_product_pt_const_znx<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_pt_const_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

// Suppress unused import warnings
#[allow(unused_imports)]
use poulpy_core::layouts::LWEInfos;
