use crate::CKKSResult as Result;
use poulpy_core::{
    DiagonalProd, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::LtDiagonalMeta,
    oep::derived::linear_transformation::{
        ckks_eval_linear_transformation_assign_derived, ckks_eval_linear_transformation_self_assign_derived,
        ckks_eval_linear_transformation_self_into_derived,
    },
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures. See `poulpy-hal/docs/backend_safety_contract.md`.
pub unsafe trait CKKSLinearTransformationImpl: Backend + super::CKKSCopyImpl {
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P: LWEInfos>(module: &Module<Self>, pt_infos: &P) -> usize;

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        key: &K,
    ) -> usize;

    fn ckks_eval_linear_transformation_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(module: &Module<Self>, ct: &C, key: &K)
    -> usize;

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        key: &K,
    ) -> usize;

    fn ckks_eval_linear_transformation_sum_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        key: &K,
    ) -> usize;

    fn ckks_prepare_linear_transformation_rhs<P>(
        module: &Module<Self>,
        prepared: &mut LinearTransformationPrepared<Self>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>;

    fn ckks_prepare_linear_transformation_baby_steps<Src, H>(
        module: &Module<Self>,
        babies: &mut LinearTransformationBabySteps<Self>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>;

    fn ckks_eval_linear_transformation_into<Dst, Src, P, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<Self>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<Self>;

    /// [`Self::ckks_eval_linear_transformation_into`] into a working copy carved from scratch, then copied back.
    fn ckks_eval_linear_transformation_assign<Dst, P, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<Self>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_eval_linear_transformation_assign_derived(module, dst, babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_sum_into<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<Self>],
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>;

    /// [`Self::ckks_eval_linear_transformation_into`] over a baby cache allocated and prepared from `src`.
    fn ckks_eval_linear_transformation_self_into<Dst, Src, P, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_eval_linear_transformation_self_into_derived(module, dst, src, lt, keys, scratch)
    }

    /// [`Self::ckks_eval_linear_transformation_self_into`] into a working copy carved from scratch, then copied
    /// back.
    fn ckks_eval_linear_transformation_self_assign<Dst, P, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_eval_linear_transformation_self_assign_derived(module, dst, lt, keys, scratch)
    }
}

/// Selects the reference implementation of `CKKSLinearTransformationImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_linear_transformation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSLinearTransformationImpl for $be {
            fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P: ::poulpy_core::layouts::LWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                pt_infos: &P,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_rhs_tmp_bytes_reference(
                    module, pt_infos,
                )
            }

            fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<
                C: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_baby_steps_tmp_bytes_reference(
                    module, ct, key,
                )
            }

            fn ckks_eval_linear_transformation_tmp_bytes<C: $crate::CKKSCtBounds, K: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_tmp_bytes_reference(
                    module, ct, key,
                )
            }

            fn ckks_eval_linear_transformation_streamed_tmp_bytes<
                C: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_streamed_tmp_bytes_reference(
                    module, ct, key,
                )
            }

            fn ckks_eval_linear_transformation_sum_tmp_bytes<
                C: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_sum_tmp_bytes_reference(
                    module, ct, key,
                )
            }

            fn ckks_prepare_linear_transformation_rhs<P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                prepared: &mut ::poulpy_core::LinearTransformationPrepared<Self>,
                lt: &::poulpy_core::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::DiagonalProd<Self>,
            {
                $crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_rhs_reference(
                    module, prepared, lt, scratch,
                )
            }

            fn ckks_prepare_linear_transformation_baby_steps<Src, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                babies: &mut ::poulpy_core::LinearTransformationBabySteps<Self>,
                src: &Src,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_baby_steps_reference(
                    module, babies, src, keys, scratch,
                )
            }

            fn ckks_eval_linear_transformation_into<Dst, Src, P, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                babies: &::poulpy_core::LinearTransformationBabySteps<Self>,
                lt: &::poulpy_core::LinearTransformation<P>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                P: ::poulpy_core::DiagonalProd<Self>
                    + $crate::api::LtDiagonalMeta
                    + ::poulpy_core::layouts::IntPolyInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_into_reference(
                    module, dst, src, babies, lt, keys, scratch,
                )
            }

            fn ckks_eval_linear_transformation_sum_into<Dst, Src, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                srcs: &[&Src],
                babies: &[&::poulpy_core::LinearTransformationBabySteps<Self>],
                lts: &[&::poulpy_core::LinearTransformationPrepared<Self>],
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_sum_into_reference(
                    module, dst, srcs, babies, lts, keys, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_linear_transformation_reference_full;
