//! The compositions of a replicated layout, the lazy dot product, the inner sum and `ckks_replicated_transform`,
//! enter the selected rounding, copy and linear transformation, are sized by their queries and match the oracle.
use super::{MAX_COPY_CAPACITY, OverrideBackend, SHAPING_CALLS};
use poulpy_ckks::{
    CKKSLayout, CKKSModuleInfos, GLWEPlaintextMeta, Scale, SlotLayout,
    api::{CKKSDotProductOps, CKKSInnerSumOps, CKKSReplicatedOps},
    layouts::ReplicatedTransform,
    shaping::ShapingUnit,
    test_suite::parity::{
        helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
        keys::{key_layout, prepared_automorphism_key, prepared_tensor_key},
    },
};
use poulpy_core::layouts::GLWELayout;
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::layouts::{Backend, Module, Standard};
use std::{cell::Cell, collections::HashMap};

thread_local! {
    static LT_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// The reference linear transformation, counting its evaluations.
unsafe impl poulpy_ckks::oep::CKKSLinearTransformationImpl<Standard> for OverrideBackend {
    fn ckks_linear_transformation_rhs_prepare_tmp_bytes<P: ::poulpy_core::layouts::GLWEInfos>(
        module: &::poulpy_hal::layouts::Module<Self>,
        pt_infos: &P,
    ) -> usize {
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_rhs_prepare_tmp_bytes_reference(
            module, pt_infos,
        )
    }

    fn ckks_linear_transformation_baby_steps_prepare_tmp_bytes<
        C: poulpy_ckks::CKKSCtBounds,
        K: ::poulpy_core::layouts::GGLWEInfos,
    >(
        module: &::poulpy_hal::layouts::Module<Self>,
        ct: &C,
        plan: &::poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize {
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_baby_steps_prepare_tmp_bytes_reference(
            module, ct, plan, key,
        )
    }

    fn ckks_linear_transformation_eval_tmp_bytes<
        C: poulpy_ckks::CKKSCtBounds,
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
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_eval_streamed_tmp_bytes<
        C: poulpy_ckks::CKKSCtBounds,
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
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_streamed_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_eval_sum_tmp_bytes<
        C: poulpy_ckks::CKKSCtBounds,
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
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_eval_sum_streamed_tmp_bytes<
        C: poulpy_ckks::CKKSCtBounds,
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
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_streamed_tmp_bytes_reference(
            module, ct, diagonal, plan, tsk, key,
        )
    }

    fn ckks_linear_transformation_rhs_prepare<P>(
        module: &::poulpy_hal::layouts::Module<Self>,
        prepared: &mut ::poulpy_core::LinearTransformationPrepared<Self, Standard>,
        lt: &::poulpy_core::LinearTransformation<P>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        P: ::poulpy_core::layouts::GLWEToBackendRef<Self, Standard> + poulpy_ckks::CKKSCtBounds,
    {
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_rhs_prepare_reference(
            module, prepared, lt, scratch,
        )
    }

    fn ckks_linear_transformation_baby_steps_prepare<Src, H, L: ::poulpy_core::layouts::BabyStepOperand<Self, Standard>>(
        module: &::poulpy_hal::layouts::Module<Self>,
        babies: &mut ::poulpy_core::LinearTransformationBabySteps<Self, Standard, L>,
        src: &Src,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Src: ::poulpy_core::layouts::GLWEToBackendRef<Self, Standard> + poulpy_ckks::CKKSCtBounds,
        H: ::poulpy_core::layouts::GetAutomorphismKey<Self, Standard>,
    {
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_baby_steps_prepare_reference(
            module, babies, src, keys, scratch,
        )
    }

    fn ckks_linear_transformation_eval_into<Dst, Src, P, H, L: ::poulpy_core::layouts::BabyStepOperand<Self, Standard>>(
        module: &::poulpy_hal::layouts::Module<Self>,
        dst: &mut Dst,
        src: &Src,
        babies: &::poulpy_core::LinearTransformationBabySteps<Self, Standard, L>,
        lt: &::poulpy_core::LinearTransformation<P>,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, Standard> + poulpy_ckks::CKKSCtBounds + poulpy_ckks::SetCKKSInfos,
        Src: ::poulpy_core::layouts::GLWEToBackendRef<Self, Standard> + poulpy_ckks::CKKSCtBounds,
        P: ::poulpy_core::GLWEDiagonalProd<Self, Standard> + poulpy_ckks::api::LtDiagonalMeta,
        H: ::poulpy_core::layouts::GetAutomorphismKey<Self, Standard> + ::poulpy_core::layouts::GetTensorKey<Self, Standard>,
    {
        LT_CALLS.with(|calls| calls.set(calls.get() + 1));
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_into_reference(
            module, dst, src, babies, lt, keys, scratch,
        )
    }

    fn ckks_linear_transformation_eval_sum_into<Dst, Src, P, H, L: ::poulpy_core::layouts::BabyStepOperand<Self, Standard>>(
        module: &::poulpy_hal::layouts::Module<Self>,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&::poulpy_core::LinearTransformationBabySteps<Self, Standard, L>],
        lts: &[&::poulpy_core::LinearTransformation<P>],
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, Standard> + poulpy_ckks::CKKSCtBounds + poulpy_ckks::SetCKKSInfos,
        Src: ::poulpy_core::layouts::GLWEToBackendRef<Self, Standard> + poulpy_ckks::CKKSCtBounds,
        P: ::poulpy_core::GLWEDiagonalProd<Self, Standard> + poulpy_ckks::api::LtDiagonalMeta,
        H: ::poulpy_core::layouts::GetAutomorphismKey<Self, Standard> + ::poulpy_core::layouts::GetTensorKey<Self, Standard>,
    {
        LT_CALLS.with(|calls| calls.set(calls.get() + 1));
        poulpy_ckks::reference::CKKSLinearTransformationReference::<Self, Standard>::ckks_linear_transformation_eval_sum_into_reference(
            module, dst, srcs, babies, lts, keys, scratch,
        )
    }
}

const N: usize = 64;
const BASE2K: usize = 12;

/// Period 16 holding 2 values, on the 32 slots of degree 64.
fn layout(k: usize) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: N.into(),
            base2k: BASE2K.into(),
            k: k.into(),
            rank: 1usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(BASE2K - 1),
            slot_layout: SlotLayout::Replicated(ShapingUnit::U16_20.layout(1).unwrap()),
            ..Default::default()
        },
    }
}

/// Counts of the shaping corrections, linear transformation evaluations and widest copy of `run`.
fn counted(run: impl FnOnce()) -> (usize, usize, usize) {
    SHAPING_CALLS.with(|calls| calls.set(0));
    LT_CALLS.with(|calls| calls.set(0));
    MAX_COPY_CAPACITY.with(|capacity| capacity.set(0));
    run();
    (
        SHAPING_CALLS.with(Cell::get),
        LT_CALLS.with(Cell::get),
        MAX_COPY_CAPACITY.with(Cell::get),
    )
}

/// The outputs of the lazy dot product, the inner sum and `ckks_replicated_transform` on `module`, into a destination
/// `dst_k` wide, each on exactly its query, with the counts of the override backend's hooks.
fn compositions<BE>(module: &Module<BE>, dst_k: usize) -> Vec<(&'static str, Snapshot, (usize, usize, usize))>
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: CKKSDotProductOps<BE>
        + CKKSInnerSumOps<BE>
        + CKKSReplicatedOps<BE>
        + CKKSModuleInfos
        + poulpy_ckks::layouts::replicated::ReplicatedTransformModule<BE, f64>
        + poulpy_core::api::GLWETensorKeyPreparedFactory<BE>
        + poulpy_core::api::GLWEAutomorphismKeyPreparedFactory<BE>
        + poulpy_core::GLWEFill<BE>,
{
    let src = layout(8 * BASE2K + 5);
    let SlotLayout::Replicated(replicated) = src.meta.slot_layout else {
        unreachable!()
    };
    let a = fixture_ciphertext::<BE, Standard>(module, &src, 71);
    let b = fixture_ciphertext::<BE, Standard>(module, &src, 73);
    let dst = layout(dst_k);
    let key = key_layout(N, BASE2K, src.glwe_layout.k.as_usize() + 64, 1, 1, 1);
    let tsk = prepared_tensor_key::<BE, Standard>(module, &key, 117);
    let mut results = Vec::new();

    let mut out = fixture_ciphertext::<BE, Standard>(module, &dst, 99);
    let counts = counted(|| {
        let bytes = module.ckks_dot_product_ct_tmp_bytes(2, &out, &a, &b, &key);
        with_scratch::<BE, _>(bytes, |s| module.ckks_dot_product_ct(&mut out, &[&a, &b], &[&b, &a], &tsk, s)).unwrap();
    });
    results.push(("dot_product_ct", snapshot::<BE, _, Standard>(&out), counts));

    let (log_batch, log_count) = (replicated.log_r(), replicated.log_k());
    let sum_keys: HashMap<i64, _> = module
        .ckks_inner_sum_galois_elements(N.into(), log_batch, log_count)
        .into_iter()
        .map(|p| {
            (
                p,
                prepared_automorphism_key::<BE, Standard>(module, &key, p, (p as u8).wrapping_add(31)),
            )
        })
        .collect();
    let mut out = fixture_ciphertext::<BE, Standard>(module, &dst, 99);
    let counts = counted(|| {
        with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&out, &a, &key), |s| {
            module.ckks_inner_sum(log_batch, log_count, false, &mut out, &a, &sum_keys, s)
        })
        .unwrap();
    });
    results.push(("inner_sum", snapshot::<BE, _, Standard>(&out), counts));

    let (k, r) = (1usize << replicated.log_k(), 1usize << replicated.log_r());
    let encode = ReplicatedTransform::<BE, Standard>::encode_tmp_bytes::<f64>(module, &replicated, BASE2K.into(), 20).unwrap();
    let transform = with_scratch::<BE, _>(encode, |s| {
        ReplicatedTransform::<BE, Standard>::from_input_period(
            module,
            replicated,
            k * r,
            BASE2K.into(),
            20,
            true,
            |s, c| {
                if s % r == 0 && c % r == 0 {
                    (((s / r + 2 * (c / r)) % 5) as f64 * 0.125 - 0.25, 0.0)
                } else {
                    (0.0, 0.0)
                }
            },
            s,
        )
    })
    .unwrap();
    let rotation_keys: HashMap<i64, _> = transform
        .rotations()
        .into_iter()
        .map(|rotation| {
            let p = module.ckks_galois_element::<Standard>(rotation);
            (
                p,
                prepared_automorphism_key::<BE, Standard>(module, &key, p, (p as u8).wrapping_add(37)),
            )
        })
        .collect();
    let mut workspace = transform.alloc_replicated_workspace(module, &a);
    let mut out = fixture_ciphertext::<BE, Standard>(module, &dst, 99);
    let counts = counted(|| {
        with_scratch::<BE, _>(module.ckks_replicated_transform_tmp_bytes(&a, &key, &transform), |s| {
            module.ckks_replicated_transform(&mut out, &a, &transform, &mut workspace, &rotation_keys, s)
        })
        .unwrap();
    });
    results.push(("replicated_transform", snapshot::<BE, _, Standard>(&out), counts));
    results
}

/// One shaped rounding each, of two columns, the wide copies of the inner sum and the transform through the selected
/// copy within their queries, and the transform's factors through the selected linear transformation: into a
/// destination as wide as the sources and into narrower ones, the narrowest taking the transform through a modulus
/// drop, bit for bit the oracle's.
#[test]
fn replicated_compositions_enter_the_selected_round_copy_and_transform() {
    let module = Module::<OverrideBackend>::new(N as u64);
    let oracle = Module::<FFT64Oracle>::new(N as u64);
    for dst_k in [8 * BASE2K + 5, 5 * BASE2K + 3, 3 * BASE2K] {
        let got = compositions(&module, dst_k);
        let want = compositions(&oracle, dst_k);
        for ((label, output, (shaping, lt, copy)), (_, expected, _)) in got.into_iter().zip(want) {
            assert_eq!(output, expected, "{label} into k = {dst_k}");
            assert_eq!(shaping, 2, "{label} into k = {dst_k}: shaping corrections");
            match label {
                "dot_product_ct" => assert_eq!((lt, copy), (0, 0), "{label}"),
                "inner_sum" => {
                    assert_eq!(lt, 0, "{label}");
                    assert!(copy > dst_k.div_ceil(BASE2K), "{label} into k = {dst_k}: the wide copy");
                }
                _ => {
                    assert!(lt > 0, "{label}: the selected linear transformation");
                    assert!(copy > (8 * BASE2K + 5).div_ceil(BASE2K), "{label}: the wide copy");
                }
            }
        }
    }
}
