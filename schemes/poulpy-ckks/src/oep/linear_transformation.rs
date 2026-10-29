use crate::CKKSResult as Result;
use poulpy_core::{
    DiagonalProd, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LinearTransformationPlan,
    },
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

/// Backend contract of [`CKKSLinearTransformationOps`](crate::api::CKKSLinearTransformationOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSLinearTransformationImpl: Backend + super::CKKSCopyImpl {
    /// Scratch bound of [`Self::ckks_prepare_linear_transformation_rhs`]; may report less than the reference, never
    /// less than the body takes.
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P: GLWEInfos>(module: &Module<Self>, pt_infos: &P) -> usize;

    /// Scratch bound of [`Self::ckks_prepare_linear_transformation_baby_steps`]; may report less than the reference,
    /// never less than the body takes.
    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize;

    /// Scratch bound of [`Self::ckks_eval_linear_transformation_into`] with a prepared (resident) right side, and of
    /// [`Self::ckks_eval_linear_transformation_assign`], whose `dst`-shaped working copy and copy back it counts; may
    /// report less than the reference, never less than the body takes.
    fn ckks_eval_linear_transformation_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize;

    /// Scratch bound of [`Self::ckks_eval_linear_transformation_into`] with a streamed (unprepared) right side, which
    /// additionally holds one resident diagonal slot, and of [`Self::ckks_eval_linear_transformation_assign`] on it,
    /// whose working copy it counts; may report less than the reference, never less than the body takes.
    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize;

    /// Scratch bound of [`Self::ckks_eval_linear_transformation_sum_into`]; may report less than the reference, never
    /// less than the body takes.
    fn ckks_eval_linear_transformation_sum_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize;

    /// Reproduces
    /// [`ckks_prepare_linear_transformation_rhs_reference`](crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_rhs_reference)
    /// bit for bit, within [`Self::ckks_prepare_linear_transformation_rhs_tmp_bytes`]; free in kernels, prepared form
    /// and scratch use. A `prepared` allocated for another layout than `lt`'s, or diagonals not sharing one scale,
    /// width and slot kind, are `InvalidOperand`, before any write.
    fn ckks_prepare_linear_transformation_rhs<P>(
        module: &Module<Self>,
        prepared: &mut LinearTransformationPrepared<Self>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_prepare_linear_transformation_baby_steps_reference`](crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_baby_steps_reference)
    /// bit for bit, within [`Self::ckks_prepare_linear_transformation_baby_steps_tmp_bytes`]; free in kernels and
    /// scratch use.
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

    /// Reproduces
    /// [`ckks_eval_linear_transformation_into_reference`](crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_into_reference)
    /// bit for bit, within [`Self::ckks_eval_linear_transformation_tmp_bytes`] for a prepared right side and
    /// [`Self::ckks_eval_linear_transformation_streamed_tmp_bytes`] for a streamed one; free in kernels and scratch
    /// use.
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
        P: DiagonalProd<Self> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>;

    /// Default body: `ckks_eval_linear_transformation_assign_derived`, [`Self::ckks_eval_linear_transformation_into`]
    /// from `dst` into a `dst`-shaped working copy carved from scratch, then the selected `ckks_copy` of the copy back
    /// into `dst`, which moves the metadata the evaluation stamped.
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
        P: DiagonalProd<Self> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>,
    {
        ckks_eval_linear_transformation_assign_derived(module, dst, babies, lt, keys, scratch)
    }

    /// Reproduces
    /// [`ckks_eval_linear_transformation_sum_into_reference`](crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_sum_into_reference)
    /// bit for bit, within [`Self::ckks_eval_linear_transformation_sum_tmp_bytes`]; free in kernels and scratch use.
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
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>;

    /// Default body: `ckks_eval_linear_transformation_self_into_derived`, a baby cache allocated for `src`,
    /// [`Self::ckks_prepare_linear_transformation_baby_steps`] into it, then
    /// [`Self::ckks_eval_linear_transformation_into`] over it. An unprepared `lt` streams its diagonals inside the
    /// evaluation. Bounded by the larger of [`Self::ckks_prepare_linear_transformation_baby_steps_tmp_bytes`] and the
    /// evaluation query of the right side, [`Self::ckks_eval_linear_transformation_tmp_bytes`] or
    /// [`Self::ckks_eval_linear_transformation_streamed_tmp_bytes`].
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
        P: DiagonalProd<Self> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>,
    {
        ckks_eval_linear_transformation_self_into_derived(module, dst, src, lt, keys, scratch)
    }

    /// Default body: `ckks_eval_linear_transformation_self_assign_derived`,
    /// [`Self::ckks_eval_linear_transformation_self_into`] from `dst` into a `dst`-shaped working copy carved from
    /// scratch, then the selected `ckks_copy` of the copy back into `dst`. Bounded by the bytes of that working copy
    /// plus the bound of [`Self::ckks_eval_linear_transformation_self_into`], whose evaluation query already counts the
    /// copy back.
    fn ckks_eval_linear_transformation_self_assign<Dst, P, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<Self> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>,
    {
        ckks_eval_linear_transformation_self_assign_derived(module, dst, lt, keys, scratch)
    }
}

/// Selects the reference implementation of `CKKSLinearTransformationImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_linear_transformation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSLinearTransformationImpl for $be {
            fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P: ::poulpy_core::layouts::GLWEInfos>(
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
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_prepare_linear_transformation_baby_steps_tmp_bytes_reference(
                    module, ct, plan, key,
                )
            }

            fn ckks_eval_linear_transformation_tmp_bytes<C: $crate::CKKSCtBounds, D: ::poulpy_core::layouts::GLWEInfos, K: ::poulpy_core::layouts::GGLWEInfos, T: ::poulpy_core::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                diagonal: &D,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                tsk: &T,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_tmp_bytes_reference(
                    module, ct, diagonal, plan, tsk, key,
                )
            }

            fn ckks_eval_linear_transformation_streamed_tmp_bytes<
                C: $crate::CKKSCtBounds,
                D: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                diagonal: &D,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                tsk: &T,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_streamed_tmp_bytes_reference(
                    module, ct, diagonal, plan, tsk, key,
                )
            }

            fn ckks_eval_linear_transformation_sum_tmp_bytes<
                C: $crate::CKKSCtBounds,
                D: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                diagonal: &D,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                tsk: &T,
                key: &K,
            ) -> usize {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_sum_tmp_bytes_reference(
                    module, ct, diagonal, plan, tsk, key,
                )
            }

            fn ckks_prepare_linear_transformation_rhs<P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                prepared: &mut ::poulpy_core::LinearTransformationPrepared<Self>,
                lt: &::poulpy_core::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
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
                P: ::poulpy_core::DiagonalProd<Self> + $crate::api::LtDiagonalMeta,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self> + ::poulpy_core::layouts::GetTensorKey<Self>,
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
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self> + ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::CKKSLinearTransformationReference::ckks_eval_linear_transformation_sum_into_reference(
                    module, dst, srcs, babies, lts, keys, scratch,
                )
            }

        }
    };
}
pub use crate::impl_ckks_linear_transformation_reference_full;
