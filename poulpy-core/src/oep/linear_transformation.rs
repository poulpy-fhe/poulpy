#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformation,
    prepared::{LinearTransformationBabySteps, PreparedDiagonal},
};

/// Backend hook for the linear-transformation family.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait LinearTransformationImpl: Backend {
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &Module<Self>, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &Module<Self>,
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

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &Module<Self>, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<Self>, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<Self>,
        prepared: &mut LinearTransformation<PreparedDiagonal<Self::OwnedBuf, Self>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        module: &Module<Self>,
        cache: &mut LinearTransformationBabySteps<Self>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>;

    fn glwe_eval_linear_transformation_into<R, P, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<Self>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: crate::reference::linear_transformation::DiagonalProd<Self>,
        H: GetAutomorphismKey<Self>;
}

/// Override surface for the linear-transformation family.
///
/// Abstract: no method bodies. See [`crate::reference::linear_transformation`]
/// for the reference algorithms a backend may forward to (the
/// [`crate::impl_linear_transformation_reference_full`] macro wires every method
/// to them).
pub trait LinearTransformationReference<BE: Backend> {
    fn glwe_eval_linear_transformation_tmp_bytes_reference<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference<R, A, B, K>(
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

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    fn glwe_prepare_linear_transformation_rhs_reference<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_prepare_linear_transformation_baby_steps_reference<A, H>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;

    fn glwe_eval_linear_transformation_into_reference<R, P, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::reference::linear_transformation::DiagonalProd<BE>,
        H: GetAutomorphismKey<BE>;
}

unsafe impl<BE> LinearTransformationImpl for BE
where
    BE: Backend,
    Module<BE>: LinearTransformationReference<BE>,
{
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &Module<BE>, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_eval_linear_transformation_tmp_bytes_reference(res, a, pt, key)
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
        module.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference(res, a, pt, key)
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &Module<BE>, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference(a, key)
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<BE>, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        module.glwe_prepare_linear_transformation_rhs_tmp_bytes_reference(pt_infos)
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<BE>,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_prepare_linear_transformation_rhs_reference(prepared, lt, scratch)
    }

    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        module: &Module<BE>,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        module.glwe_prepare_linear_transformation_baby_steps_reference(cache, a, keys, scratch)
    }

    fn glwe_eval_linear_transformation_into<R, P, H>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::reference::linear_transformation::DiagonalProd<BE>,
        H: GetAutomorphismKey<BE>,
    {
        module.glwe_eval_linear_transformation_into_reference(cnv_offset, res, lhs, rhs, keys, scratch)
    }
}

/// Implements [`LinearTransformationReference`] for `Module<$be>` by forwarding
/// every method to the corresponding `crate::reference::linear_transformation`
/// reference function.
///
/// For partial override (custom kernel for one method, defaults for the rest),
/// write the impl block by hand and forward only the methods you keep.
#[macro_export]
macro_rules! impl_linear_transformation_reference_full {
    ($be:ty) => {
        impl $crate::oep::LinearTransformationReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_eval_linear_transformation_tmp_bytes_reference<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference::<$be, _, _, _, _, _>(
                    self, res, a, pt, key,
                )
            }

            fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference<R, A, B, K>(
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
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference::<
                    $be,
                    _,
                    _,
                    _,
                    _,
                    _,
                >(self, res, a, pt, key)
            }

            fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference<A, K>(&self, a: &A, key: &K) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference::<
                    $be,
                    _,
                    _,
                    _,
                >(self, a, key)
            }

            fn glwe_prepare_linear_transformation_rhs_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
            where
                P: $crate::layouts::LWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference::<$be, _, _>(
                    self, pt_infos,
                )
            }

            fn glwe_prepare_linear_transformation_rhs_reference<P>(
                &self,
                prepared: &mut $crate::layouts::LinearTransformation<
                    $crate::layouts::prepared::PreparedDiagonal<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                >,
                lt: &$crate::layouts::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference::<$be, _, _>(
                    self, prepared, lt, scratch,
                )
            }

            fn glwe_prepare_linear_transformation_baby_steps_reference<A, H>(
                &self,
                cache: &mut $crate::layouts::prepared::LinearTransformationBabySteps<$be>,
                a: &A,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference::<$be, _, _, _>(
                    self, cache, a, keys, scratch,
                )
            }

            fn glwe_eval_linear_transformation_into_reference<R, P, H>(
                &self,
                cnv_offset: usize,
                res: &mut R,
                lhs: &$crate::layouts::prepared::LinearTransformationBabySteps<$be>,
                rhs: &$crate::layouts::LinearTransformation<P>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                P: $crate::reference::linear_transformation::DiagonalProd<$be>,
                H: $crate::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_into_reference::<$be, _, _, _, _>(
                    self, cnv_offset, res, lhs, rhs, keys, scratch,
                )
            }
        }
    };
}
