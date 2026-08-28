#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::GLWELinearTransformations,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement, LWEInfos, LinearTransformation,
        prepared::{GGLWEPreparedToBackendRef, LinearTransformationBabySteps, PreparedDiagonal},
    },
    oep::LinearTransformationImpl,
};

impl<BE> GLWELinearTransformations<BE> for Module<BE>
where
    BE: Backend + LinearTransformationImpl<BE>,
{
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, P, H, K>(
        &self,
        res: &R,
        a: &A,
        rhs: &LinearTransformation<P>,
        keys: &H,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        P: LWEInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        BE::glwe_eval_linear_transformation_tmp_bytes(self, res, a, rhs, keys)
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, P, H, K>(
        &self,
        res: &R,
        a: &A,
        rhs: &LinearTransformation<P>,
        keys: &H,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        P: LWEInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        BE::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(self, res, a, rhs, keys)
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, H, K>(&self, a: &A, rotations: &[i64], keys: &H) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        BE::glwe_prepare_linear_transformation_baby_steps_tmp_bytes(self, a, rotations, keys)
    }

    fn glwe_eval_linear_transformation_bound_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_eval_linear_transformation_bound_tmp_bytes(self, res, a, pt, key)
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_bound_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_eval_linear_transformation_unprepared_rhs_bound_tmp_bytes(self, res, a, pt, key)
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        BE::glwe_prepare_linear_transformation_rhs_tmp_bytes(self, pt_infos)
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_prepare_linear_transformation_rhs(self, prepared, lt, scratch)
    }

    fn glwe_prepare_linear_transformation_baby_steps<A, H, K>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        BE::glwe_prepare_linear_transformation_baby_steps(self, cache, a, keys, scratch)
    }

    fn glwe_eval_linear_transformation_into<R, P, H, K>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        BE::glwe_eval_linear_transformation_into(self, cnv_offset, res, lhs, rhs, keys, scratch)
    }
}
