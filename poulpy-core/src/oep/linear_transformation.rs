#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
    LinearTransformation,
    prepared::{GGLWEPreparedToBackendRef, LinearTransformationBabySteps, PreparedDiagonal},
};

/// Backend hook for the linear-transformation family.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
#[allow(private_bounds)]
pub unsafe trait LinearTransformationImpl<BE: Backend>: Backend {
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &Module<BE>, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        pt: &B,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &Module<BE>, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<BE>, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<BE>,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps<A, H, K>(
        module: &Module<BE>,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        a_k: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_eval_linear_transformation_into<R, P, H, K>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

/// Override surface for the linear-transformation family.
///
/// Abstract: no method bodies. See [`crate::default::linear_transformation`]
/// for the reference algorithms a backend may forward to (the
/// [`crate::impl_linear_transformation_defaults_full`] macro wires every method
/// to them).
#[doc(hidden)]
#[allow(private_bounds)]
pub trait LinearTransformationDefault<BE: Backend> {
    fn glwe_eval_linear_transformation_tmp_bytes_default<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default<R, A, B, K>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes_default<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    fn glwe_prepare_linear_transformation_rhs_default<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps_default<A, H, K>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        a_k: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_eval_linear_transformation_into_default<R, P, H, K>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

#[allow(private_bounds)]
unsafe impl<BE> LinearTransformationImpl<BE> for BE
where
    BE: Backend,
    Module<BE>: LinearTransformationDefault<BE>,
{
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &Module<BE>, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_eval_linear_transformation_tmp_bytes_default(res, a, pt, key)
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        pt: &B,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default(res, a, pt, key)
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &Module<BE>, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default(a, key)
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<BE>, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        module.glwe_prepare_linear_transformation_rhs_tmp_bytes_default(pt_infos)
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<BE>,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_prepare_linear_transformation_rhs_default(prepared, lt, scratch)
    }

    fn glwe_prepare_linear_transformation_baby_steps<A, H, K>(
        module: &Module<BE>,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        a_k: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.glwe_prepare_linear_transformation_baby_steps_default(cache, a, a_k, keys, key_size, scratch)
    }

    fn glwe_eval_linear_transformation_into<R, P, H, K>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.glwe_eval_linear_transformation_into_default(cnv_offset, res, lhs, rhs, keys, key_size, scratch)
    }
}

/// Implements [`LinearTransformationDefault`] for `Module<$be>` by forwarding
/// every method to the corresponding `crate::default::linear_transformation`
/// reference function.
///
/// For partial override (custom kernel for one method, defaults for the rest),
/// write the impl block by hand and forward only the methods you keep.
#[macro_export]
macro_rules! impl_linear_transformation_defaults_full {
    ($be:ty) => {
        impl $crate::oep::LinearTransformationDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_eval_linear_transformation_tmp_bytes_default<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_default::<$be, _, _, _, _, _>(
                    self, res, a, pt, key,
                )
            }

            fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default<R, A, B, K>(
                &self,
                res: &R,
                a: &A,
                pt: &B,
                key: &K,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default::<
                    $be,
                    _,
                    _,
                    _,
                    _,
                    _,
                >(self, res, a, pt, key)
            }

            fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default<A, K>(&self, a: &A, key: &K) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default::<
                    $be,
                    _,
                    _,
                    _,
                >(self, a, key)
            }

            fn glwe_prepare_linear_transformation_rhs_tmp_bytes_default<P>(&self, pt_infos: &P) -> usize
            where
                P: $crate::layouts::LWEInfos,
            {
                $crate::default::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_default::<$be, _, _>(
                    self, pt_infos,
                )
            }

            fn glwe_prepare_linear_transformation_rhs_default<P>(
                &self,
                prepared: &mut $crate::layouts::LinearTransformation<
                    $crate::layouts::prepared::PreparedDiagonal<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                >,
                lt: &$crate::layouts::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::linear_transformation::glwe_prepare_linear_transformation_rhs_default::<$be, _, _>(
                    self, prepared, lt, scratch,
                )
            }

            fn glwe_prepare_linear_transformation_baby_steps_default<A, H, K>(
                &self,
                cache: &mut $crate::layouts::prepared::LinearTransformationBabySteps<$be>,
                a: &A,
                a_k: usize,
                keys: &H,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::GetGaloisElement
                    + $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be>
                    + $crate::layouts::GGLWEInfos,
                H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::linear_transformation::glwe_prepare_linear_transformation_baby_steps_default::<$be, _, _, _, _>(
                    self, cache, a, a_k, keys, key_size, scratch,
                )
            }

            fn glwe_eval_linear_transformation_into_default<R, P, H, K>(
                &self,
                cnv_offset: usize,
                res: &mut R,
                lhs: &$crate::layouts::prepared::LinearTransformationBabySteps<$be>,
                rhs: &$crate::layouts::LinearTransformation<P>,
                keys: &H,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                P: $crate::default::linear_transformation::DiagonalProd<$be>,
                K: $crate::layouts::GetGaloisElement
                    + $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be>
                    + $crate::layouts::GGLWEInfos,
                H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::linear_transformation::glwe_eval_linear_transformation_into_default::<$be, _, _, _, _, _>(
                    self, cnv_offset, res, lhs, rhs, keys, key_size, scratch,
                )
            }
        }
    };
}
