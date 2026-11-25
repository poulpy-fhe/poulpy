//! The replicated conversions, whose derived bodies reach the linear-transformation and inner-sum hooks, counted on
//! the standard ring, each behind a marked extra its query reports.
use super::{COPY_CALLS, OverrideBackend, ROUND_CALLS, ROUND_EXTRA, ROUND_EXTRA_DEFAULT, SHAPING_CALLS, carry};
use poulpy_ckks::api::{CKKSReplicatedOps, LtDiagonalMeta};
use poulpy_ckks::layouts::ReplicatedConversionPrepared;
use poulpy_ckks::oep::{CKKSInnerSumImpl, CKKSLinearTransformationImpl};
use poulpy_ckks::reference::CKKSLinearTransformationReference;
use poulpy_ckks::reference::inner_sum::CKKSInnerSumReference;
use poulpy_ckks::shaping::ShapingUnit;
use poulpy_ckks::test_suite::helpers::assert_invalid_operand;
use poulpy_ckks::test_suite::parity::helpers::with_scratch;
use poulpy_ckks::test_suite::parity::{keys::key_layout, layout, run_refused_replicated_conversions, run_replicated_conversions};
use poulpy_ckks::{CKKSCtBounds, CKKSLayout, CKKSResult, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind};
use poulpy_core::layouts::{
    BabyStepOperand, Degree, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
    LinearTransformationPlan,
};
use poulpy_core::{GLWEDiagonalProd, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared};
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::layouts::{Module, ScratchArena, Standard};
use std::cell::Cell;
thread_local! {
    static LT_CALLS: Cell<usize> = const { Cell::new(0) };
    static LT_EXTRA: Cell<usize> = const { Cell::new(0) };
    static INNER_SUM_CALLS: Cell<usize> = const { Cell::new(0) };
    static INNER_SUM_EXTRA: Cell<usize> = const { Cell::new(0) };
}
const LT_EXTRA_RUN: usize = 1088;
const INNER_SUM_EXTRA_RUN: usize = 1152;

unsafe impl CKKSLinearTransformationImpl<Standard> for OverrideBackend {
    fn ckks_linear_transformation_rhs_prepare_tmp_bytes<P: GLWEInfos>(module: &Module<Self>, pt_infos: &P) -> usize {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_rhs_prepare_tmp_bytes_reference(
            module, pt_infos,
        )
    }

    fn ckks_linear_transformation_baby_steps_prepare_tmp_bytes<C: CKKSCtBounds, K: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_baby_steps_prepare_tmp_bytes_reference(
            module, ct, plan, key,
        )
    }

    fn ckks_linear_transformation_eval_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize {
        LT_EXTRA.get()
            + CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_tmp_bytes_reference(
                module, ct, diagonal, plan, tsk, key,
            )
    }

    fn ckks_linear_transformation_eval_streamed_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize {
        LT_EXTRA.get()
            + CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_streamed_tmp_bytes_reference(
                module, ct, diagonal, plan, tsk, key,
            )
    }

    fn ckks_linear_transformation_eval_sum_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_eval_sum_streamed_tmp_bytes<C: CKKSCtBounds, D: GLWEInfos, K: GGLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_streamed_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_rhs_prepare<P>(
        module: &Module<Self>,
        prepared: &mut LinearTransformationPrepared<Self, Standard>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        P: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
    {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_rhs_prepare_reference(
            module, prepared, lt, scratch,
        )
    }

    fn ckks_linear_transformation_baby_steps_prepare<Src, H, L: BabyStepOperand<Self, Standard>>(
        module: &Module<Self>,
        babies: &mut LinearTransformationBabySteps<Self, Standard, L>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Src: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Standard>,
    {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_baby_steps_prepare_reference(
            module, babies, src, keys, scratch,
        )
    }

    fn ckks_linear_transformation_eval_into<Dst, Src, P, H, L: BabyStepOperand<Self, Standard>>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<Self, Standard, L>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
        P: GLWEDiagonalProd<Self, Standard> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self, Standard> + GetTensorKey<Self, Standard>,
    {
        LT_CALLS.set(LT_CALLS.get() + 1);
        carry::with_marked_bytes(LT_EXTRA.get(), scratch, |rest| {
            CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_into_reference(
                module, dst, src, babies, lt, keys, rest,
            )
        })
    }

    fn ckks_linear_transformation_eval_sum_into<Dst, Src, P, H, L: BabyStepOperand<Self, Standard>>(
        module: &Module<Self>,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<Self, Standard, L>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
        P: GLWEDiagonalProd<Self, Standard> + LtDiagonalMeta,
        H: GetAutomorphismKey<Self, Standard> + GetTensorKey<Self, Standard>,
    {
        CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_into_reference(
            module, dst, srcs, babies, lts, keys, scratch,
        )
    }
}

unsafe impl CKKSInnerSumImpl<Standard> for OverrideBackend {
    fn ckks_inner_sum_galois_elements(module: &Module<Self>, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64> {
        CKKSInnerSumReference::<Self, Standard>::ckks_inner_sum_galois_elements_reference(module, n, log_batch, log_count)
    }

    fn ckks_replicate_galois_elements(module: &Module<Self>, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64> {
        CKKSInnerSumReference::<Self, Standard>::ckks_replicate_galois_elements_reference(module, n, log_batch, log_count)
    }

    fn ckks_inner_sum_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        INNER_SUM_EXTRA.get() + CKKSInnerSumReference::<Self, Standard>::ckks_inner_sum_tmp_bytes_reference(module, res, a, key)
    }

    fn ckks_inner_sum<R, A, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Standard>,
    {
        CKKSInnerSumReference::<Self, Standard>::ckks_inner_sum_reference(
            module, log_batch, log_count, normalize, res, a, keys, scratch,
        )
    }

    fn ckks_inner_sum_assign<R, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self, Standard>,
    {
        INNER_SUM_CALLS.set(INNER_SUM_CALLS.get() + 1);
        carry::with_marked_bytes(INNER_SUM_EXTRA.get(), scratch, |rest| {
            CKKSInnerSumReference::<Self, Standard>::ckks_inner_sum_assign_reference(
                module, log_batch, log_count, normalize, res, keys, rest,
            )
        })
    }

    fn ckks_replicate<R, A, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Standard> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Standard>,
    {
        CKKSInnerSumReference::<Self, Standard>::ckks_replicate_reference(module, log_batch, log_count, res, a, keys, scratch)
    }

    fn ckks_replicate_assign<R, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self, Standard> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self, Standard>,
    {
        CKKSInnerSumReference::<Self, Standard>::ckks_replicate_assign_reference(module, log_batch, log_count, res, keys, scratch)
    }
}

/// The conversions of the replicated parity, each on exactly its query, enter the round hook onto the replicated layout,
/// the linear-transformation hook per factor, the inner-sum hook back and the copy hook on both maps, refused ones none,
/// and give the oracle's outputs; the conversion query takes the extras of the round, linear-transformation and
/// inner-sum hooks.
#[test]
fn replicated_conversions_enter_their_hooks() {
    let params = poulpy_ckks::test_suite::CKKSTestParams {
        n: 64,
        hw: 48,
        base2k: 12,
        ..poulpy_ckks::test_suite::BASE19_PARAMS_F64
    };
    let module = Module::<OverrideBackend>::new(64);
    let unit = ShapingUnit::U8_40.layout(2).unwrap();
    let base2k = params.base2k.into();
    let bytes = ReplicatedConversionPrepared::<OverrideBackend>::new_tmp_bytes::<f64>(&module, &unit, base2k, 30).unwrap();
    let conv = with_scratch::<OverrideBackend, _>(bytes, |s| {
        ReplicatedConversionPrepared::<_, Standard>::new::<f64>(&module, 64usize.into(), unit, base2k, 20, 30, 17, s)
    })
    .unwrap();
    let a = layout(params, 1, 120, 40, 0, SlotsKind::Complex);
    let res = CKKSLayout {
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(20),
            slot_layout: SlotLayout::Replicated(unit),
            ..a.meta
        },
        ..a
    };
    let key = key_layout(64, 12, 160, 1, 1, 1);
    let query = || module.ckks_replicated_conversion_tmp_bytes(&res, &a, &conv, &key);
    carry::assert_query_takes_extra("ckks_replicated_conversion_tmp_bytes", |bytes| ROUND_EXTRA.set(bytes), query);
    ROUND_EXTRA.set(ROUND_EXTRA_DEFAULT);
    carry::assert_query_takes_extra("ckks_replicated_conversion_tmp_bytes", |bytes| LT_EXTRA.set(bytes), query);
    carry::assert_query_takes_extra(
        "ckks_replicated_conversion_tmp_bytes",
        |bytes| INNER_SUM_EXTRA.set(bytes),
        query,
    );
    LT_EXTRA.set(LT_EXTRA_RUN);
    INNER_SUM_EXTRA.set(INNER_SUM_EXTRA_RUN);

    let counters = [&ROUND_CALLS, &SHAPING_CALLS, &COPY_CALLS, &LT_CALLS, &INNER_SUM_CALLS];
    let counts = || counters.map(|calls| calls.get());
    counters.iter().for_each(|calls| calls.set(0));
    let got = run_replicated_conversions::<OverrideBackend, f64, Standard>(params, &module);
    // Three sources, dense and factored, each with two conversions onto the replicated layout and two back. Onto it, a
    // round with a shaping correction per half; every conversion copies into its wide stage and once per factor of its
    // map, one dense and two factored, each factor one evaluation, and the narrow one onto the layout and both back once
    // more, out of the round or the trace; back, one trace each.
    assert_eq!(counts(), [12, 24, 78, 36, 12]);
    counters.iter().for_each(|calls| calls.set(0));
    let refused = run_refused_replicated_conversions::<OverrideBackend, f64, Standard>(params, &module);
    assert_eq!(counts(), [0; 5], "a refused conversion entered a hook");
    for (error, op) in refused.iter().zip(["ckks_to_replicated", "ckks_from_replicated"]) {
        assert_invalid_operand(op, error, op, "of one rank");
    }
    let oracle = Module::<FFT64Oracle>::new(64);
    assert_eq!(
        got,
        run_replicated_conversions::<FFT64Oracle, f64, Standard>(params, &oracle),
        "the probe's conversions differ from the oracle's"
    );
    LT_EXTRA.set(0);
    INNER_SUM_EXTRA.set(0);
}
