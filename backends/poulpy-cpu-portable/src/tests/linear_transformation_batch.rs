//! The batched giant-step products of a linear transformation, on a backend whose
//! `glwe_giant_step_batch` is above 1, so that the batched paths of the evaluation and their
//! scratch sizing run in CI: the parity suites against `FFT64Portable`, on the exact budgets
//! the queries return.

use crate::{FFT64Portable, hal_impl::delegating_backend::BatchedGiantStepsFFT64Portable as Batched};
use poulpy_core::test_suite::parity::{
    ParityShapes, test_linear_transformation_parity, test_linear_transformation_powers_of_two_parity,
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut},
    test_suite::TestParams,
};
use std::cell::Cell;

/// The giant steps the fixture's products are computed in: not a divisor of the giant-step
/// counts of the suites, so that a batch is also cut short.
const BATCH: usize = 3;

thread_local! {
    /// Calls of the products of more than one giant step on this thread, which runs a test and
    /// its serial evaluations.
    static BATCHED_CALLS: Cell<usize> = const { Cell::new(0) };
}

crate::impl_sampling_host!(Batched, fft64);
poulpy_core::impl_ggsw_conversion_reference_full!(Batched);
poulpy_core::impl_decryption_reference_full!(Batched);
poulpy_core::impl_encryption_reference_full!(Batched);
poulpy_core::impl_operations_reference_full!(Batched);
poulpy_core::impl_polynomial_evaluation_derived_full!(Batched);
poulpy_core::impl_preparation_reference_full!(Batched);
poulpy_core::impl_secret_reference_full!(Batched);
poulpy_core::impl_gglwe_external_product_derived_full!(Batched);
poulpy_core::impl_gglwe_keyswitch_derived_full!(Batched);
poulpy_core::impl_ggsw_external_product_derived_full!(Batched);
poulpy_core::impl_ggsw_keyswitch_derived_full!(Batched);
poulpy_core::impl_automorphism_reference_full!(Batched);
poulpy_core::impl_glwe_external_product_reference_full!(Batched);
poulpy_core::impl_glwe_keyswitch_reference_full!(Batched);
poulpy_core::impl_lwe_keyswitch_reference_full!(Batched);
poulpy_core::impl_glwe_tensoring_reference_full!(Batched);
poulpy_core::impl_gglwe_product_digits_strided_reference_full!(Batched);
poulpy_core::impl_glwe_trace_derived_full!(Batched);

unsafe impl poulpy_core::oep::LinearTransformationImpl for Batched {
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_sum_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        a: &A,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference(
            module, a, plan, key,
        )
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &::poulpy_hal::layouts::Module<Batched>, pt_infos: &P) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference(
            module, pt_infos,
        )
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        prepared: &mut poulpy_core::layouts::LinearTransformation<
            poulpy_core::layouts::prepared::PreparedDiagonal<<Batched as ::poulpy_hal::layouts::Backend>::OwnedBuf, Batched>,
        >,
        lt: &poulpy_core::layouts::LinearTransformation<P>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<Batched>,
    ) where
        P: poulpy_core::layouts::GLWEToBackendRef<Batched> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference(
            module, prepared, lt, scratch,
        )
    }

    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        cache: &mut poulpy_core::layouts::prepared::LinearTransformationBabySteps<Batched>,
        a: &A,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<Batched>,
    ) where
        A: poulpy_core::layouts::GLWEToBackendRef<Batched> + poulpy_core::layouts::GLWEInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<Batched>,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference(
            module, cache, a, keys, scratch,
        )
    }

    fn glwe_eval_linear_transformation_into<R, P, H>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<Batched>,
        rhs: &poulpy_core::layouts::LinearTransformation<P>,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<Batched>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<Batched> + poulpy_core::layouts::GLWEInfos,
        P: poulpy_core::layouts::DiagonalProd<Batched>,
        H: poulpy_core::layouts::GetAutomorphismKey<Batched>,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_into_reference(
            module, cnv_offset, res, lhs, rhs, keys, scratch,
        )
    }

    fn glwe_eval_linear_transformation_sum_into<R, H>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&poulpy_core::layouts::prepared::LinearTransformationBabySteps<Batched>],
        lts: &[&poulpy_core::layouts::prepared::LinearTransformationPrepared<Batched>],
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<Batched>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<Batched> + poulpy_core::layouts::GLWEInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<Batched>,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference(
            module, cnv_offset, res, babies, lts, keys, scratch,
        )
    }

    fn glwe_eval_ct_linear_transformation_tmp_bytes<R, A, B, K, T>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        res: &R,
        a: &A,
        b: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
        tsk: &T,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
        T: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_ct_linear_transformation_tmp_bytes_reference(
            module, res, a, b, plan, key, tsk,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_ct_linear_transformation_into<R, D, T, H>(
        module: &::poulpy_hal::layouts::Module<Batched>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<Batched>,
        lhs_k: poulpy_core::layouts::TorusPrecision,
        rhs: &poulpy_core::layouts::LinearTransformation<D>,
        tsk: &T,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Batched>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<Batched> + poulpy_core::layouts::GLWEInfos,
        D: ::poulpy_hal::layouts::CnvPVecRToBackendRef<Batched> + poulpy_core::layouts::GLWEInfos,
        T: poulpy_core::layouts::GetTensorKey<Batched>,
        H: poulpy_core::layouts::GetAutomorphismKey<Batched>,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_ct_linear_transformation_into_reference(
            module, cnv_offset, res, lhs, lhs_k, rhs, tsk, keys, scratch,
        )
    }

    fn glwe_giant_step_batch(_module: &Module<Self>) -> usize {
        BATCH
    }

    fn glwe_accumulate_giant_steps(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, Self>],
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<Self>,
        steps: &[poulpy_core::layouts::LinearTransformationGiantStep<
            poulpy_core::layouts::prepared::PreparedDiagonal<<Self as Backend>::OwnedBuf, Self>,
        >],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert!(prods.len() <= BATCH, "more giant steps than the batch");
        if prods.len() > 1 {
            BATCHED_CALLS.set(BATCHED_CALLS.get() + 1);
        }
        poulpy_core::reference::linear_transformation::glwe_accumulate_giant_steps_reference(
            module,
            cnv_offset_hi,
            prods,
            lhs,
            steps,
            scratch,
        )
    }
}

fn run(test: fn(&TestParams, &ParityShapes, &Module<FFT64Portable>, &Module<Batched>)) {
    let params = TestParams {
        size: 1 << 8,
        base2k: 12,
        n: 1 << 8,
    };
    let before = BATCHED_CALLS.get();
    test(
        &params,
        &ParityShapes::default(),
        &Module::<FFT64Portable>::new(params.size as u64),
        &Module::<Batched>::new(params.size as u64),
    );
    assert!(BATCHED_CALLS.get() > before, "no product of several giant steps ran");
}

#[test]
fn batched_linear_transformation_parity() {
    run(test_linear_transformation_parity::<FFT64Portable, Batched>);
}

#[test]
fn batched_linear_transformation_powers_of_two_parity() {
    run(test_linear_transformation_powers_of_two_parity::<FFT64Portable, Batched>);
}
