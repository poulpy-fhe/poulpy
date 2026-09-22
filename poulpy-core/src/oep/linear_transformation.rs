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

/// Implements the linear-transformation backend hooks with HAL-based algorithms.
///
/// Each method can instead be implemented directly by the backend. The reference
/// functions remain callable independently for methods whose algorithms are reused.
#[macro_export]
macro_rules! impl_linear_transformation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::LinearTransformationImpl for $be {
            fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, pt: &B, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference::<$be, _, _, _, _, _>(
                    module, res, a, pt, key,
                )
            }

            fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
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
                >(module, res, a, pt, key)
            }

            fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &::poulpy_hal::layouts::Module<$be>, a: &A, key: &K) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference::<
                    $be,
                    _,
                    _,
                    _,
                >(module, a, key)
            }

            fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &::poulpy_hal::layouts::Module<$be>, pt_infos: &P) -> usize
            where
                P: $crate::layouts::LWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference::<$be, _, _>(
                    module, pt_infos,
                )
            }

            fn glwe_prepare_linear_transformation_rhs<P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                prepared: &mut $crate::layouts::LinearTransformation<
                    $crate::layouts::prepared::PreparedDiagonal<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                >,
                lt: &$crate::layouts::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference::<$be, _, _>(
                    module, prepared, lt, scratch,
                )
            }

            fn glwe_prepare_linear_transformation_baby_steps<A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cache: &mut $crate::layouts::prepared::LinearTransformationBabySteps<$be>,
                a: &A,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference::<$be, _, _, _>(
                    module, cache, a, keys, scratch,
                )
            }

            fn glwe_eval_linear_transformation_into<R, P, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
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
                    module, cnv_offset, res, lhs, rhs, keys, scratch,
                )
            }
        }
    };
}
