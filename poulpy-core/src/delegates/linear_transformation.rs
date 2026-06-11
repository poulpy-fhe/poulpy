#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::GLWELinearTransformations,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, LinearTransformation, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement, LWEInfos,
        prepared::{GGLWEPreparedToBackendRef, LinearTransformationLhsPrepared, LinearTransformationRhsPrepared},
    },
    oep::LinearTransformationImpl,
};

impl<BE> GLWELinearTransformations<BE> for Module<BE>
where
    BE: Backend + LinearTransformationImpl<BE>,
{
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_eval_linear_transformation_tmp_bytes(self, res, a, pt, key)
    }

    fn glwe_prepare_linear_transformation_lhs_tmp_bytes<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_prepare_linear_transformation_lhs_tmp_bytes(self, a, key)
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        BE::glwe_prepare_linear_transformation_rhs_tmp_bytes(self, pt_infos)
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationRhsPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_prepare_linear_transformation_rhs(self, prepared, lt, scratch)
    }

    fn glwe_prepare_linear_transformation_lhs<A, H, K>(
        &self,
        cache: &mut LinearTransformationLhsPrepared<BE>,
        a: &A,
        a_effective_k: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::glwe_prepare_linear_transformation_lhs(self, cache, a, a_effective_k, key_size, keys, scratch)
    }

    fn glwe_eval_linear_transformation_into<R, H, K>(
        &self,
        res: &mut R,
        lhs: &LinearTransformationLhsPrepared<BE>,
        rhs: &LinearTransformationRhsPrepared<BE>,
        cnv_offset: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::glwe_eval_linear_transformation_into(self, res, lhs, rhs, cnv_offset, key_size, keys, scratch)
    }
}
