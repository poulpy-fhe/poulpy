use crate::CKKSResult as Result;
use poulpy_core::{
    LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LWEInfos},
    reference::linear_transformation::DiagonalProd,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos, SlotLayout,
    api::{CKKSLinearTransformationOps, LtDiagonalMeta},
    oep::CKKSLinearTransformationImpl,
};

impl<BE: Backend + CKKSLinearTransformationImpl> CKKSLinearTransformationOps<BE> for Module<BE> {
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        BE::ckks_prepare_linear_transformation_rhs_tmp_bytes_impl(self, pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_prepare_linear_transformation_baby_steps_tmp_bytes_impl(self, ct, key)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_eval_linear_transformation_tmp_bytes_impl(self, ct, key)
    }

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_eval_linear_transformation_streamed_tmp_bytes_impl(self, ct, key)
    }

    fn ckks_eval_linear_transformation_sum_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_eval_linear_transformation_sum_tmp_bytes_impl(self, ct, key)
    }

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        BE::ckks_prepare_linear_transformation_rhs_impl(self, prepared, lt, scratch)
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
        crate::reference::mul::ensure_full_layout(
            "ckks_prepare_linear_transformation_baby_steps",
            crate::GLWEPlaintextInfos::slot_layout(src),
        )?;
        BE::ckks_prepare_linear_transformation_baby_steps_impl(self, babies, src, keys, scratch)
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
        H: GetAutomorphismKey<BE>,
    {
        crate::reference::mul::ensure_full_layout(
            "ckks_eval_linear_transformation_into",
            crate::GLWEPlaintextInfos::slot_layout(src),
        )?;
        BE::ckks_eval_linear_transformation_into_impl(self, dst, src, babies, lt, keys, scratch)?;
        dst.set_slot_layout(SlotLayout::Full);
        Ok(())
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
        H: GetAutomorphismKey<BE>,
    {
        crate::reference::mul::ensure_full_layout(
            "ckks_eval_linear_transformation_assign",
            crate::GLWEPlaintextInfos::slot_layout(dst),
        )?;
        BE::ckks_eval_linear_transformation_assign_impl(self, dst, babies, lt, keys, scratch)
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
        H: GetAutomorphismKey<BE>,
    {
        for src in srcs {
            crate::reference::mul::ensure_full_layout(
                "ckks_eval_linear_transformation_sum_into",
                crate::GLWEPlaintextInfos::slot_layout(*src),
            )?;
        }
        BE::ckks_eval_linear_transformation_sum_into_impl(self, dst, srcs, babies, lts, keys, scratch)?;
        dst.set_slot_layout(SlotLayout::Full);
        Ok(())
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
        H: GetAutomorphismKey<BE>,
    {
        crate::reference::mul::ensure_full_layout(
            "ckks_eval_linear_transformation_self_into",
            crate::GLWEPlaintextInfos::slot_layout(src),
        )?;
        BE::ckks_eval_linear_transformation_self_into_impl(self, dst, src, lt, keys, scratch)?;
        dst.set_slot_layout(SlotLayout::Full);
        Ok(())
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
        H: GetAutomorphismKey<BE>,
    {
        crate::reference::mul::ensure_full_layout(
            "ckks_eval_linear_transformation_self_assign",
            crate::GLWEPlaintextInfos::slot_layout(dst),
        )?;
        BE::ckks_eval_linear_transformation_self_assign_impl(self, dst, lt, keys, scratch)
    }
}
