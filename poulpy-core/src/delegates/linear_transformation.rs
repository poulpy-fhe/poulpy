#![allow(clippy::too_many_arguments)]

use crate::layouts::BabyStepOperand;
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    api::GLWELinearTransformationOps,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LinearTransformation,
        LinearTransformationPlan,
        prepared::{LinearTransformationBabySteps, PreparedDiagonal},
    },
    oep::GLWELinearTransformationImpl,
};

impl<BE, Rg: Ring> GLWELinearTransformationOps<BE, Rg> for Module<BE>
where
    BE: Backend + GLWELinearTransformationImpl<Rg>,
{
    fn glwe_linear_transformation_eval_tmp_bytes<R, A, B, K, T>(
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
        BE::glwe_linear_transformation_eval_tmp_bytes(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes<R, A, B, K, T>(
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
        BE::glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_sum_tmp_bytes<R, A, B, K, T>(
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
        BE::glwe_linear_transformation_eval_sum_tmp_bytes(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes<R, A, B, K, T>(
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
        BE::glwe_linear_transformation_eval_sum_unprepared_rhs_tmp_bytes(self, res, a, pt, plan, tsk, key)
    }

    fn glwe_linear_transformation_baby_steps_prepare_tmp_bytes<A, K>(
        &self,
        a: &A,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_linear_transformation_baby_steps_prepare_tmp_bytes(self, a, plan, key)
    }

    fn glwe_linear_transformation_rhs_prepare_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: GLWEInfos,
    {
        BE::glwe_linear_transformation_rhs_prepare_tmp_bytes(self, pt_infos)
    }

    fn glwe_linear_transformation_rhs_prepare<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE, Rg>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_linear_transformation_rhs_prepare(self, prepared, lt, scratch)
    }

    fn glwe_linear_transformation_baby_steps_prepare<A, H, L: BabyStepOperand<BE, Rg>>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE, Rg, L>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        H: GetAutomorphismKey<BE, Rg>,
    {
        BE::glwe_linear_transformation_baby_steps_prepare(self, cache, a, keys, scratch)
    }

    fn glwe_linear_transformation_eval_into<R, P, H, L: BabyStepOperand<BE, Rg>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE, Rg, L>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        P: crate::layouts::GLWEDiagonalProd<BE, Rg>,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        BE::glwe_linear_transformation_eval_into(self, cnv_offset, res, lhs, rhs, keys, scratch)
    }

    fn glwe_linear_transformation_eval_sum_into<R, P, H, L: BabyStepOperand<BE, Rg>>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&LinearTransformationBabySteps<BE, Rg, L>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        P: crate::layouts::GLWEDiagonalProd<BE, Rg>,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        BE::glwe_linear_transformation_eval_sum_into(self, cnv_offset, res, babies, lts, keys, scratch)
    }
}
