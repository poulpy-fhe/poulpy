use crate::CKKSResult as Result;
use poulpy_core::{
    DiagonalProd, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos,
        LinearTransformationPlan,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, LtDiagonalMeta},
    oep::CKKSLinearTransformationImpl,
};

impl<BE: Backend + CKKSLinearTransformationImpl> CKKSLinearTransformationOps<BE> for Module<BE> {
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: GLWEInfos,
    {
        BE::ckks_prepare_linear_transformation_rhs_tmp_bytes(self, pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(
        &self,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_prepare_linear_transformation_baby_steps_tmp_bytes(self, ct, plan, key)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, D, K, T>(
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
        BE::ckks_eval_linear_transformation_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, K>(&self, ct: &C, plan: &LinearTransformationPlan, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_eval_linear_transformation_streamed_tmp_bytes(self, ct, plan, key)
    }

    fn ckks_eval_linear_transformation_sum_tmp_bytes<C, D, K, T>(
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
        BE::ckks_eval_linear_transformation_sum_tmp_bytes(self, ct, diagonal, plan, tsk, key)
    }

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_linear_transformation_rhs(self, prepared, lt, scratch)
    }

    fn ckks_prepare_linear_transformation_baby_steps<Src, H>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_prepare_linear_transformation_baby_steps(self, babies, src, keys, scratch)
    }

    fn ckks_eval_linear_transformation_into<Dst, Src, P, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_eval_linear_transformation_into(self, dst, src, babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_eval_linear_transformation_assign(self, dst, babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_sum_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<BE>],
        lts: &[&LinearTransformationPrepared<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_eval_linear_transformation_sum_into(self, dst, srcs, babies, lts, keys, scratch)
    }

    fn ckks_eval_linear_transformation_self_into<Dst, Src, P, H>(
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
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_eval_linear_transformation_self_into(self, dst, src, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_self_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_eval_linear_transformation_self_assign(self, dst, lt, keys, scratch)
    }
}
