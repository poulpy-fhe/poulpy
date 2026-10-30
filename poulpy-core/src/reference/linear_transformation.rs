//! GLWE-level linear transformation (matrix-vector product over the slots) in
//! baby-step / giant-step (BSGS) form.
//!
//! This module is scheme-agnostic: CKKS derives scale/capacity metadata and
//! passes only base2k alignment integers (`a_k`, `cnv_offset`). The
//! files under `linear_transformation/` follow the phases in `docs/guides/linear_transformation.md`:
//! schedule construction, setup/preparation, baby hoisting, giant-step products
//! and rotations, and final normalization.

mod baby_steps;
mod ciphertext_diagonals;
mod eval;
mod giant_steps;
mod inner_product;
mod lazy;
mod prepare;

#[cfg(test)]
mod tests;

// The data types and BSGS schedule derivation live in `crate::layouts`; the
// prepared (convolution-domain) caches live in `crate::layouts::prepared`.
// This module owns the HAL-dependent allocators and the prepare/eval
// algorithms. Re-exported here so `poulpy_core::*` keeps exposing them at the
// crate root.
pub use crate::layouts::prepared::{LinearTransformationBabySteps, LinearTransformationPrepared, PreparedDiagonal};
pub use crate::layouts::{
    BsgsRotations, LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationLayout,
    LinearTransformationPlan, LinearTransformationStrategy, optimal_bsgs_giant_step,
};

// Forwarding targets of `GLWELinearTransformationReference`.
pub use eval::{
    glwe_linear_transformation_baby_steps_prepare_reference, glwe_linear_transformation_baby_steps_prepare_tmp_bytes_reference,
    glwe_linear_transformation_eval_into_reference, glwe_linear_transformation_eval_sum_into_reference,
    glwe_linear_transformation_eval_sum_tmp_bytes_reference,
    glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes_reference, glwe_linear_transformation_eval_tmp_bytes_reference,
    glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes_reference,
};
pub use inner_product::{glwe_accumulate_giant_steps_reference, glwe_giant_step_batch_reference};
pub use prepare::{glwe_linear_transformation_rhs_prepare_reference, glwe_linear_transformation_rhs_prepare_tmp_bytes_reference};

use poulpy_hal::{
    api::{
        CnvOps, CnvPVecAlloc, CnvPVecBytesOf, ModuleN, VecZnxAutomorphism, VecZnxAutomorphismAssign, VecZnxBigAddAssign,
        VecZnxBigAlloc, VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxDftAutomorphism, VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero,
        VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize,
        VecZnxIdftNormalizeTmpBytes, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, GaloisElement, Module, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    api::{GLWEAdd, GLWEAutomorphism, GLWEBytesOf, GLWECopy, GLWEMulPlain, GLWENormalize, GLWEPreparedTensoring},
    layouts::{
        BabyStepOperand, GGLWEInfos, GLWEDiagonalProd, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
        GetTensorKey, ModuleCoreAlloc,
    },
    oep::{GGLWEProductDigitsStridedImpl, GLWELinearTransformationImpl, GLWETensoringImpl},
    reference::{
        keyswitching::{GGLWEProductReference, GLWEKeyswitchInternal},
        operations::GLWETensoringReference,
    },
};

/// Reference bodies of the [`GLWELinearTransformationImpl`] methods, which
/// [`impl_glwe_linear_transformation_reference_full!`](crate::impl_glwe_linear_transformation_reference_full) forwards to.
pub trait GLWELinearTransformationReference<BE: Backend> {
    fn glwe_linear_transformation_eval_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn glwe_linear_transformation_eval_sum_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn glwe_linear_transformation_baby_steps_prepare_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_linear_transformation_rhs_prepare_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        P: GLWEInfos;

    fn glwe_linear_transformation_rhs_prepare_reference<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_linear_transformation_baby_steps_prepare_reference<A, H, L: BabyStepOperand<BE>>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE, L>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;

    fn glwe_linear_transformation_eval_into_reference<R, P, H, L: BabyStepOperand<BE>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE, L>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEDiagonalProd<BE>,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    fn glwe_linear_transformation_eval_sum_into_reference<R, P, H, L: BabyStepOperand<BE>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&LinearTransformationBabySteps<BE, L>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEDiagonalProd<BE>,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    fn glwe_giant_step_batch_reference(&self) -> usize;

    fn glwe_accumulate_giant_steps_reference<L: BabyStepOperand<BE>>(
        &self,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &LinearTransformationBabySteps<BE, L>,
        steps: &[LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>],
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend> GLWELinearTransformationReference<BE> for Module<BE>
where
    BE: GGLWEProductDigitsStridedImpl + GLWELinearTransformationImpl + GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + GLWEPreparedTensoring<BE>
        + GLWETensoringReference<BE>
        + GLWENormalize<BE>
        + GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulPlain<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ModuleN
        + CnvOps<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf
        + GGLWEProductReference<BE>
        + GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphism<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + GaloisElement
        + Sync,
{
    fn glwe_linear_transformation_eval_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        eval::glwe_linear_transformation_eval_tmp_bytes_reference(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        eval::glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes_reference(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_sum_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        eval::glwe_linear_transformation_eval_sum_tmp_bytes_reference(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes_reference<R, A, B, K, T>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        eval::glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes_reference(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_baby_steps_prepare_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        eval::glwe_linear_transformation_baby_steps_prepare_tmp_bytes_reference(self, a, plan, key)
    }

    fn glwe_linear_transformation_rhs_prepare_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        P: GLWEInfos,
    {
        prepare::glwe_linear_transformation_rhs_prepare_tmp_bytes_reference(self, pt_infos)
    }

    fn glwe_linear_transformation_rhs_prepare_reference<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        prepare::glwe_linear_transformation_rhs_prepare_reference(self, prepared, lt, scratch)
    }

    fn glwe_linear_transformation_baby_steps_prepare_reference<A, H, L: BabyStepOperand<BE>>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE, L>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        eval::glwe_linear_transformation_baby_steps_prepare_reference(self, cache, a, keys, scratch)
    }

    fn glwe_linear_transformation_eval_into_reference<R, P, H, L: BabyStepOperand<BE>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE, L>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEDiagonalProd<BE>,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        eval::glwe_linear_transformation_eval_into_reference(self, cnv_offset, res, lhs, rhs, keys, scratch)
    }

    fn glwe_linear_transformation_eval_sum_into_reference<R, P, H, L: BabyStepOperand<BE>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&LinearTransformationBabySteps<BE, L>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEDiagonalProd<BE>,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        eval::glwe_linear_transformation_eval_sum_into_reference(self, cnv_offset, res, babies, lts, keys, scratch)
    }

    fn glwe_giant_step_batch_reference(&self) -> usize {
        inner_product::glwe_giant_step_batch_reference(self)
    }

    fn glwe_accumulate_giant_steps_reference<L: BabyStepOperand<BE>>(
        &self,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &LinearTransformationBabySteps<BE, L>,
        steps: &[LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>],
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        inner_product::glwe_accumulate_giant_steps_reference(self, cnv_offset_hi, prods, lhs, steps, scratch)
    }
}
