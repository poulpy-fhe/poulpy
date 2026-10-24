#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut};

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformation,
    LinearTransformationGiantStep, LinearTransformationPlan,
    prepared::{LinearTransformationBabySteps, LinearTransformationPrepared, PreparedDiagonal},
};

/// Backend hook for the linear-transformation family.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait LinearTransformationImpl: Backend {
    /// Scratch bound of [`Self::glwe_eval_linear_transformation_into`]; may report less than the reference, never
    /// less than the body takes.
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bound of [`Self::glwe_eval_linear_transformation_into`] when the right-hand side is streamed, an
    /// unprepared plaintext diagonal; may report less than the reference, never less than the body takes.
    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bound of [`Self::glwe_eval_linear_transformation_sum_into`]; may report less than the reference, never
    /// less than the body takes.
    fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bound of [`Self::glwe_prepare_linear_transformation_baby_steps`]; may report less than the reference,
    /// never less than the body takes.
    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(
        module: &Module<Self>,
        a: &A,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bound of [`Self::glwe_prepare_linear_transformation_rhs`]; may report less than the reference, never
    /// less than the body takes.
    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<Self>, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    /// Reproduces
    /// [`glwe_prepare_linear_transformation_rhs_reference`](crate::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_prepare_linear_transformation_rhs_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<Self>,
        prepared: &mut LinearTransformation<PreparedDiagonal<Self::OwnedBuf, Self>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_prepare_linear_transformation_baby_steps_reference`](crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_prepare_linear_transformation_baby_steps_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        module: &Module<Self>,
        cache: &mut LinearTransformationBabySteps<Self>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>;

    /// Reproduces
    /// [`glwe_eval_linear_transformation_into_reference`](crate::reference::linear_transformation::glwe_eval_linear_transformation_into_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_eval_linear_transformation_tmp_bytes`], or
    /// [`Self::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes`] for a streamed right-hand side; free in
    /// kernels and scratch use.
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
        P: crate::layouts::DiagonalProd<Self>,
        H: GetAutomorphismKey<Self>;

    /// Reproduces
    /// [`glwe_eval_linear_transformation_sum_into_reference`](crate::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_eval_linear_transformation_sum_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_eval_linear_transformation_sum_into<R, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&LinearTransformationBabySteps<Self>],
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>;

    /// Most giant steps [`Self::glwe_accumulate_giant_steps`] takes per call: the evaluation
    /// holds that many products at once. The default is one, the reference schedule.
    fn glwe_giant_step_batch(_module: &Module<Self>) -> usize {
        1
    }

    /// The products of several giant steps of a prepared transformation:
    /// `prods[g] = sum_i baby(steps[g].diagonals[i].baby) * steps[g].diagonals[i].plaintext`,
    /// every limb of every column of each product being written.
    ///
    /// `prods` and `steps` have the same length, at most [`Self::glwe_giant_step_batch`]. The
    /// default accumulates each step with `cnv_apply_dft_sum`. Overrides must reproduce it bit for bit.
    /// Scratch is the per-step product scratch of [`Self::glwe_eval_linear_transformation_tmp_bytes`],
    /// which reserves the `prods` of a full batch; an internal hook, it records no metadata.
    fn glwe_accumulate_giant_steps(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, Self>],
        lhs: &LinearTransformationBabySteps<Self>,
        steps: &[LinearTransformationGiantStep<PreparedDiagonal<Self::OwnedBuf, Self>>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: poulpy_hal::api::Convolution<Self>,
    {
        crate::reference::linear_transformation::glwe_accumulate_giant_steps_reference(
            module,
            cnv_offset_hi,
            prods,
            lhs,
            steps,
            scratch,
        )
    }
}

/// Implements the linear-transformation backend hooks with HAL-based algorithms.
///
/// Each method can instead be implemented directly by the backend. The reference
/// functions remain callable independently for methods whose algorithms are reused.
#[macro_export]
macro_rules! impl_linear_transformation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::LinearTransformationImpl for $be {
            fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, pt: &B, plan: &$crate::layouts::LinearTransformationPlan, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference::<$be, _, _, _, _, _>(
                    module, res, a, pt, plan, key,
                )
            }

            fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                pt: &B,
                plan: &$crate::layouts::LinearTransformationPlan,
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
                >(module, res, a, pt, plan, key)
            }

            fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, pt: &B, plan: &$crate::layouts::LinearTransformationPlan, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_sum_tmp_bytes_reference::<$be, _, _, _, _, _>(
                    module, res, a, pt, plan, key,
                )
            }

            fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(module: &::poulpy_hal::layouts::Module<$be>, a: &A, plan: &$crate::layouts::LinearTransformationPlan, key: &K) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference::<
                    $be,
                    _,
                    _,
                    _,
                >(module, a, plan, key)
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
                P: $crate::DiagonalProd<$be>,
                H: $crate::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_into_reference::<$be, _, _, _, _>(
                    module, cnv_offset, res, lhs, rhs, keys, scratch,
                )
            }

            fn glwe_eval_linear_transformation_sum_into<R, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                babies: &[&$crate::layouts::prepared::LinearTransformationBabySteps<$be>],
                lts: &[&$crate::layouts::prepared::LinearTransformationPrepared<$be>],
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference::<$be, _, _, _>(
                    module, cnv_offset, res, babies, lts, keys, scratch,
                )
            }
        }
    };
}
