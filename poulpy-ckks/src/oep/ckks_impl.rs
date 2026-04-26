#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENegate, GLWERotate, GLWEShift, GLWESub, GLWETensoring,
    ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWEPlaintext,
        GLWETensorKeyPrepared, GetGaloisElement,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
};

/// Backend-owned CKKS leveled-operations extension point.
///
/// `Module<BE>` remains the public execution surface. Backend crates can
/// implement this trait on their backend marker type to override CKKS-level
/// algorithms while preserving the existing module-facing API.
///
/// # Safety
/// Implementors must preserve all CKKS metadata invariants and must obey the
/// scratch, sizing, aliasing, and layout contracts required by the underlying
/// `poulpy-core` and `poulpy-hal` operations they call.
pub unsafe trait CKKSImpl<BE: Backend>: Backend {
    fn ckks_extract_pt_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<S: CKKSInfos>(
        module: &Module<BE>,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Module<BE>: VecZnxLsh<BE> + VecZnxRsh<BE>;

    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_vec_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add_pt_const_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_const_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_const_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_add_pt_const_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_vec_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_sub_pt_const_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_const_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_const_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_pt_const_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_neg_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWENegate + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_neg_assign(module: &Module<BE>, dst: &mut CKKSCiphertext<impl DataMut>) -> Result<()>
    where
        Module<BE>: GLWENegate;

    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_mul_pow2_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_mul_pow2_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_div_pow2_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_assign(module: &Module<BE>, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()>;

    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_rotate_into<H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_rotate_assign<H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_conjugate_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_mul_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        Module<BE>: GLWETensoring<BE>;

    fn ckks_square_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        Module<BE>: GLWETensoring<BE>;

    fn ckks_mul_pt_vec_znx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulPlain<BE>;

    fn ckks_mul_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>;

    fn ckks_mul_pt_const_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_znx_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_znx_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_rnx_into<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_mul_pt_const_rnx_assign<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

#[macro_export]
macro_rules! impl_ckks_default_methods {
    ($backend:ty) => {
        $crate::impl_ckks_pt_znx_default_methods!($backend);
        $crate::impl_ckks_add_default_methods!($backend);
        $crate::impl_ckks_sub_default_methods!($backend);
        $crate::impl_ckks_neg_default_methods!($backend);
        $crate::impl_ckks_pow2_default_methods!($backend);
        $crate::impl_ckks_rotate_default_methods!($backend);
        $crate::impl_ckks_conjugate_default_methods!($backend);
        $crate::impl_ckks_mul_default_methods!($backend);
    };
}

pub use crate::impl_ckks_default_methods;
