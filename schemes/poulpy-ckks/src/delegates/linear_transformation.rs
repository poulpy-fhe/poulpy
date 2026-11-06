use crate::CKKSResult as Result;
use poulpy_core::layouts::BabyStepOperand;
use poulpy_core::{
    GLWEDiagonalProd, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LinearTransformationPlan,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, LtDiagonalMeta},
    oep::CKKSLinearTransformationImpl,
};

impl<BE: Backend + CKKSLinearTransformationImpl> CKKSLinearTransformationOps<BE> for Module<BE> {
    fn ckks_linear_transformation_rhs_prepare_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: GLWEInfos,
    {
        BE::ckks_linear_transformation_rhs_prepare_tmp_bytes(self, pt_infos)
    }

    fn ckks_linear_transformation_baby_steps_prepare_tmp_bytes<C, K>(
        &self,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_linear_transformation_baby_steps_prepare_tmp_bytes(self, ct, plan, key)
    }

    fn ckks_linear_transformation_eval_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ckks_linear_transformation_eval_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_linear_transformation_eval_streamed_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ckks_linear_transformation_eval_streamed_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_linear_transformation_eval_self_tmp_bytes<C, D, K, T>(
        &self,
        streamed: bool,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ckks_linear_transformation_eval_self_tmp_bytes(self, streamed, ct, diagonal, plan, tsk, key)
    }

    fn ckks_linear_transformation_eval_sum_streamed_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ckks_linear_transformation_eval_sum_streamed_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_linear_transformation_eval_sum_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ckks_linear_transformation_eval_sum_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_linear_transformation_rhs_prepare<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_linear_transformation_rhs_prepare(self, prepared, lt, scratch)
    }

    fn ckks_linear_transformation_baby_steps_prepare<Src, H, L: BabyStepOperand<BE>>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE, L>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_linear_transformation_baby_steps_prepare(self, babies, src, keys, scratch)
    }

    fn ckks_linear_transformation_eval_into<Dst, Src, P, H, L: BabyStepOperand<BE>>(
        &self,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE, L>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_linear_transformation_eval_into(self, dst, src, babies, lt, keys, scratch)
    }

    fn ckks_linear_transformation_eval_assign<Dst, P, H, L: BabyStepOperand<BE>>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE, L>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_linear_transformation_eval_assign(self, dst, babies, lt, keys, scratch)
    }

    fn ckks_linear_transformation_eval_sum_into<Dst, Src, P, H, L: BabyStepOperand<BE>>(
        &self,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<BE, L>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_linear_transformation_eval_sum_into(self, dst, srcs, babies, lts, keys, scratch)
    }

    fn ckks_linear_transformation_eval_self_into<Dst, Src, P, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_linear_transformation_eval_self_into(self, dst, src, lt, keys, scratch)
    }

    fn ckks_linear_transformation_eval_self_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_linear_transformation_eval_self_assign(self, dst, lt, keys, scratch)
    }
}
