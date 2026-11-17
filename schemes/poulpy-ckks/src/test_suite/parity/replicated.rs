//! Shaped operations of a replicated slot layout: rounding, encoding, products,
//! automorphisms and polynomial evaluation with masked constants. The shaped
//! encryption runs in [`test_encryption_parity`](super::test_encryption_parity).
use super::{
    arithmetic::layout,
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, fixture_operand, fixture_plaintext, snapshot, with_scratch},
    keys::{key_layout, key_layout_bits, prepared_automorphism_key, prepared_tensor_key},
};
use crate::{
    CKKSLayout, CKKSModuleInfos, CoeffsMeta, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSEncodingHostOps, CKKSEncodingScalar, CKKSMulOps,
        CKKSPolynomialEvaluationOps, CKKSReplicatedOps, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSPlaintextAlloc, ReplicatedConversion, ReplicatedTransform, encode_bsgs_replicated,
        replicated::ReplicatedTransformModule,
    },
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSDotProductImpl, CKKSEncodingImpl, CKKSMulImpl, CKKSPlaintextVecImpl,
        CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRotateImpl, CKKSRoundImpl, CKKSSubImpl,
    },
    polynomial::{Basis, Polynomial, SplitStrategy},
    reference::encoding::CKKSSlotEmbedding,
    shaping::{ShapingProfile, ShapingUnit, conjugate_invariant},
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEAdd, GLWEAutomorphismKeyPreparedFactory, GLWEFill, GLWETensorKeyPreparedFactory,
    layouts::{GLWEAutomorphismKeyPrepared, GLWELayout, LWEInfos},
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, Ring, ScratchArena, Standard},
};
use std::collections::HashMap;

/// The CKKS families of a ring that the shaped operations use.
pub trait ReplicatedRingImpl<Rg: Ring>:
    CKKSPlaintextVecImpl<Rg>
    + CKKSCopyImpl<Rg>
    + CKKSAddImpl<Rg>
    + CKKSSubImpl<Rg>
    + CKKSRoundImpl<Rg>
    + CKKSPow2Impl<Rg>
    + CKKSRotateImpl<Rg>
    + CKKSMulImpl<Rg>
    + CKKSDotProductImpl<Rg>
    + CKKSPolynomialEvaluationImpl<Rg>
{
}

impl<B, Rg: Ring> ReplicatedRingImpl<Rg> for B where
    B: CKKSPlaintextVecImpl<Rg>
        + CKKSCopyImpl<Rg>
        + CKKSAddImpl<Rg>
        + CKKSSubImpl<Rg>
        + CKKSRoundImpl<Rg>
        + CKKSPow2Impl<Rg>
        + CKKSRotateImpl<Rg>
        + CKKSMulImpl<Rg>
        + CKKSDotProductImpl<Rg>
        + CKKSPolynomialEvaluationImpl<Rg>
{
}

/// `ckks_to_replicated` and back of a full-layout fixture of `k = 4` values onto `U8_40` (`R = 4`), with the maps
/// dense and with two placement factors, each within its queried arena, the keys fixtures at the suite's rank.
fn conversions<B, F, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSReplicatedOps<B, Rg>
        + ReplicatedTransformModule<B, F, Rg>
        + CnvPVecAlloc<B>
        + GLWEAutomorphismKeyPreparedFactory<B, Rg>
        + GLWEFill<B, Rg>,
{
    let b = params.base2k;
    let (log_k, log_delta_ckks, log_delta, log_delta_diag) = (2u8, 40, 20, 30);
    let layout = ShapingUnit::U8_40.layout(log_k).unwrap();
    let extra_bits =
        log_delta_ckks - log_delta - (module.ckks_slots_of_degree::<Rg>(params.n) >> log_k).trailing_zeros() as usize;
    let inner_log_delta_diag = log_delta + extra_bits + 4;
    let bytes =
        ReplicatedConversion::<B, Rg>::new_tmp_bytes::<F>(module, &layout, b.into(), log_delta_diag.max(inner_log_delta_diag))
            .unwrap();
    let mut results = Vec::new();
    for (label, placement_factors) in [("dense", None), ("factored", Some(2))] {
        let conv = with_scratch::<B, _>(bytes, |s| match placement_factors {
            None => ReplicatedConversion::<B, Rg>::new::<F>(
                module,
                params.n.into(),
                layout,
                b.into(),
                log_delta,
                log_delta_diag,
                extra_bits,
                s,
            ),
            Some(placement_factors) => ReplicatedConversion::<B, Rg>::new_factored::<F>(
                module,
                params.n.into(),
                layout,
                b.into(),
                log_delta,
                log_delta_diag,
                inner_log_delta_diag,
                extra_bits,
                placement_factors,
                s,
            ),
        })
        .unwrap();
        let from_bits = conv.from.as_ref().map_or(0, ReplicatedTransform::consumed_bits);
        let k = log_delta_ckks + conv.to.consumed_bits() + from_bits + log_delta_diag;
        let (_, atk_layout) = rank_keys(params, k + extra_bits + conv.trace.len() + b);
        let keys: HashMap<i64, _> = conv
            .rotations()
            .into_iter()
            .map(|r| {
                let p = crate::operand_rotation_galois_element::<_, Rg>("parity", module, params.n, r).unwrap();
                (
                    p,
                    prepared_automorphism_key::<B, Rg>(module, &atk_layout, p, (p as u8).wrapping_add(31)),
                )
            })
            .collect();
        let a = fixture_ciphertext::<B, Rg>(
            module,
            &super::arithmetic::layout(params, params.rank, k, log_delta_ckks, 0, SlotsKind::Complex),
            81,
        );
        let mut replicated =
            fixture_ciphertext::<B, Rg>(module, &ct_layout(params, k, log_delta, SlotLayout::Replicated(layout)), 83);
        let mut workspace = conv.alloc_to_workspace(module, &a);
        with_scratch::<B, _>(
            module.ckks_replicated_conversion_tmp_bytes(&replicated, &a, &atk_layout, &conv),
            |s| module.ckks_to_replicated(&mut replicated, &a, &conv, &mut workspace, &keys, s),
        )
        .unwrap();
        results.push((label, snapshot::<B, _, Rg>(&replicated)));
        let mut back = fixture_ciphertext::<B, Rg>(
            module,
            &super::arithmetic::layout(params, params.rank, k, log_delta_ckks, 0, SlotsKind::Complex),
            85,
        );
        let mut workspace = conv.alloc_from_workspace(module, &replicated).unwrap();
        with_scratch::<B, _>(
            module.ckks_replicated_conversion_tmp_bytes(&back, &replicated, &atk_layout, &conv),
            |s| module.ckks_from_replicated(&mut back, &replicated, &conv, &mut workspace, &keys, s),
        )
        .unwrap();
        results.push((label, snapshot::<B, _, Rg>(&back)));
    }
    results
}

/// The replicated layout of the suite: period 8 holding 2 values, on `2kR = 32` slots.
fn replicated() -> SlotLayout {
    SlotLayout::Replicated(ShapingUnit::U16_20.layout(1).unwrap())
}

/// A ciphertext layout of the suite's rank at width `k` and scale `2^delta`, in `slot_layout`.
fn ct_layout(params: CKKSTestParams, k: usize, delta: usize, slot_layout: SlotLayout) -> CKKSLayout {
    let mut out = layout(
        params,
        params.rank,
        k,
        delta,
        params.n.ilog2() as usize - 5,
        SlotsKind::Complex,
    );
    out.meta.slot_layout = slot_layout;
    out
}

/// `ckks_round` of `a` into a fixture of `out_layout`, within its exact queried arena.
fn round<B, Rg: Ring>(module: &Module<B>, out_layout: &CKKSLayout, a: &crate::layouts::CKKSCiphertextOwned<B, Rg>) -> Snapshot
where
    B: Backend<ZnxWord = i64> + CKKSRoundImpl<Rg>,
    Module<B>: GLWEFill<B, Rg>,
{
    let before = snapshot::<B, _, Rg>(a);
    let mut out = fixture_ciphertext::<B, Rg>(module, out_layout, 99);
    with_scratch::<B, _>(CKKSRoundOps::<B, Rg>::ckks_round_tmp_bytes(module, &out, a), |s| {
        module.ckks_round(&mut out, a, s)
    })
    .unwrap();
    assert_eq!(before, snapshot::<B, _, Rg>(a), "ckks_round changed its source");
    snapshot::<B, _, Rg>(&out)
}

/// A ciphertext layout of the suite's rank at width `k`, in the replicated layout.
fn replicated_at(params: CKKSTestParams, k: usize) -> CKKSLayout {
    ct_layout(params, k, params.base2k - 1, replicated())
}

/// Tensor and automorphism key layouts of the suite's rank at width `k`.
fn rank_keys(params: CKKSTestParams, k: usize) -> (poulpy_core::layouts::GGLWELayout, poulpy_core::layouts::GGLWELayout) {
    let (b, r) = (params.base2k, params.rank);
    (
        key_layout(params.n, b, k, 1, r * (r + 1) / 2, r),
        key_layout(params.n, b, k, 1, r, r),
    )
}

/// The roundings and the shaped encoding of the ring `Rg`.
fn rounding_ops<B, F, Rg>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSRoundImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
    Rg: CKKSSlotEmbedding,
    Module<B>: GLWEFill<B, Rg> + GLWEAdd<B, Rg>,
{
    let b = params.base2k;
    let k = 8 * b + 5;
    let mut results = Vec::new();
    for (label, slot_layout) in [("replicated", replicated()), ("full", SlotLayout::Full)] {
        let la = ct_layout(params, k, b - 1, slot_layout);
        let a = fixture_ciphertext::<B, Rg>(module, &la, 71);
        let half = ct_layout(params, k - b / 2, b - 1, slot_layout);
        results.push((label, round(module, &half, &a)));

        // A source whose digits leave the canonical range, its flag clear.
        let lazy = fixture_operand::<B, Rg>(module, &la, 73, true);
        results.push((label, round(module, &half, &lazy)));

        // The destination at another radix.
        let mut other_radix = half;
        other_radix.glwe_layout.base2k = (b - 1).into();
        results.push((label, round(module, &other_radix, &a)));
    }

    // A drop past the largest pass rounds plainly first, then with shaping.
    let SlotLayout::Replicated(unit) = replicated() else {
        unreachable!()
    };
    let pass = ShapingProfile::<Rg>::try_new(&unit).unwrap().max_pass_bits();
    let wide = ct_layout(params, k, k - b, replicated());
    let a = fixture_ciphertext::<B, Rg>(module, &wide, 75);
    let past = ct_layout(params, k - pass - 6, k - b - pass - 6, replicated());
    results.push(("replicated past the pass", round(module, &past, &a)));

    // The shaped encoding of a batch of 2 values.
    let slots = module.ckks_max_slots::<Rg>();
    // A few bits of budget, so that the shaping pass fits `i128` at every radix.
    let mut pt = module.ckks_plaintext_alloc_from_slots::<Rg>(slots, b.into(), 48usize.into());
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(40),
        slot_layout: replicated(),
        ..Default::default()
    });
    let value = |x: f64| F::from_f64(x).unwrap();
    let im = if conjugate_invariant::<Rg>() { 0.0 } else { -0.25 };
    let (re, im) = ([value(0.625), value(-0.375)], [value(im), value(-im)]);
    with_scratch::<B, _>(module.ckks_encoding_host_tmp_bytes(slots), |s| {
        module.ckks_encode_slots_host_into(&mut pt, &re, &im, s)
    })
    .unwrap();
    results.push(("replicated encoding", snapshot::<B, _, Rg>(&pt)));
    results
}

/// The shaped products, automorphisms, masked constants and polynomial
/// evaluation of the ring `Rg`.
fn shaped_ops<B, F, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + ReplicatedRingImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
    Module<B>:
        CKKSAllOpsTmpBytes<B> + GLWETensorKeyPreparedFactory<B, Rg> + GLWEAutomorphismKeyPreparedFactory<B, Rg> + GLWEFill<B, Rg>,
{
    let b = params.base2k;
    let la = replicated_at(params, 8 * b + 5);
    let lr = replicated_at(params, 5 * b + 3);
    let a = fixture_ciphertext::<B, Rg>(module, &la, 71);
    let rhs = fixture_ciphertext::<B, Rg>(module, &la, 73);
    let before = snapshot::<B, _, Rg>(&a);
    let (tsk_layout, atk_layout) = rank_keys(params, la.k().as_usize());
    let tsk = prepared_tensor_key::<B, Rg>(module, &tsk_layout, 113);
    let shift = 8;
    let rot = prepared_automorphism_key::<B, Rg>(module, &atk_layout, module.ckks_galois_element::<Rg>(shift), 62);
    let mut results = Vec::new();

    let mut out = fixture_ciphertext::<B, Rg>(module, &lr, 99);
    with_scratch::<B, _>(
        CKKSMulOps::<B, Rg>::ckks_mul_tmp_bytes(module, &out, &a, &rhs, &tsk_layout),
        |s| module.ckks_mul_into(&mut out, &a, &rhs, &tsk, s),
    )
    .unwrap();
    results.push(("mul_into", snapshot::<B, _, Rg>(&out)));

    let mut out = fixture_ciphertext::<B, Rg>(module, &lr, 99);
    with_scratch::<B, _>(
        CKKSMulOps::<B, Rg>::ckks_square_tmp_bytes(module, &out, &a, &tsk_layout),
        |s| module.ckks_square_into(&mut out, &a, &tsk, s),
    )
    .unwrap();
    results.push(("square_into", snapshot::<B, _, Rg>(&out)));

    let mut lp = replicated_at(params, b + 3);
    lp.glwe_layout = GLWELayout {
        n: 32usize.into(),
        rank: 0usize.into(),
        ..lp.glwe_layout
    };
    lp.meta.log_sparsity = 0;
    let pt = fixture_plaintext::<B, Rg>(module, &lp, 54);
    let mut out = fixture_ciphertext::<B, Rg>(module, &lr, 99);
    with_scratch::<B, _>(CKKSMulOps::<B, Rg>::ckks_mul_pt_vec_tmp_bytes(module, &out, &a, &pt), |s| {
        module.ckks_mul_pt_vec_into(&mut out, &a, &pt, s)
    })
    .unwrap();
    results.push(("mul_pt_vec_into", snapshot::<B, _, Rg>(&out)));

    let mut out = fixture_ciphertext::<B, Rg>(module, &lr, 99);
    with_scratch::<B, _>(CKKSRotateOps::<B, Rg>::ckks_rotate_tmp_bytes(module, &a, &atk_layout), |s| {
        module.ckks_rotate_into(shift, &mut out, &a, &rot, s)
    })
    .unwrap();
    results.push(("rotate_into", snapshot::<B, _, Rg>(&out)));

    narrow_automorphisms::<B, Rg>(
        params,
        module,
        module.ckks_galois_element::<Rg>(8),
        &mut results,
        |module, dst, src, key, scratch| module.ckks_rotate_into(8, dst, src, key, scratch),
    );

    for (label, add) in [("add_one_assign", true), ("sub_one_assign", false)] {
        let mut out = fixture_ciphertext::<B, Rg>(module, &la, 71);
        let bytes = if add {
            CKKSAddOps::<B, Rg>::ckks_add_one_tmp_bytes(module, out.size())
        } else {
            CKKSSubOps::<B, Rg>::ckks_sub_one_tmp_bytes(module, out.size())
        };
        with_scratch::<B, _>(bytes, |s| {
            if add {
                module.ckks_add_one_assign(&mut out, s)
            } else {
                module.ckks_sub_one_assign(&mut out, s)
            }
        })
        .unwrap();
        results.push((label, snapshot::<B, _, Rg>(&out)));
    }

    // Polynomial evaluation takes rank-1 operands, as its one-pass products do.
    if params.rank == 1 {
        let coeff_meta = CoeffsMeta::from_delta_budget(b - 1, 4);
        let polynomial = Polynomial::new(
            Basis::Monomial,
            [0.125, -0.25, 0.0625, 0.125]
                .into_iter()
                .map(|v| F::from_f64(v).unwrap())
                .collect(),
        );
        let SlotLayout::Replicated(rl) = la.meta.slot_layout else {
            unreachable!()
        };
        let encoding = <Module<B> as CKKSEncodingHostOps<B, F, Rg>>::ckks_encoding_host_tmp_bytes(module, module.n());
        let bsgs = with_scratch::<B, _>(encoding, |s| {
            encode_bsgs_replicated::<B, F, Rg>(&polynomial, module, b.into(), coeff_meta, SplitStrategy::MinDepth, rl, s)
        })
        .unwrap();
        let pt_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                k: coeff_meta.k,
                rank: 0usize.into(),
                ..la.glwe_layout
            },
            meta: coeff_meta.meta,
        };
        let mut out = fixture_ciphertext::<B, Rg>(module, &la, 99);
        with_scratch::<B, _>(module.ckks_all_ops_tmp_bytes(&la, &pt_layout, &tsk_layout), |s| {
            module.ckks_eval_poly_real_const_coeffs(&mut out, &a, &bsgs, &tsk, s)
        })
        .unwrap();
        results.push(("eval_poly", snapshot::<B, _, Rg>(&out)));
    }

    assert_eq!(before, snapshot::<B, _, Rg>(&a));
    results
}

type Automorphism<B, Rg> = fn(
    &Module<B>,
    &mut CKKSCiphertextOwned<B, Rg>,
    &CKKSCiphertextOwned<B, Rg>,
    &GLWEAutomorphismKeyPrepared<<B as Backend>::OwnedBuf, B, Rg>,
    &mut ScratchArena<'_, B>,
) -> crate::CKKSResult<()>;

/// The automorphism `apply` of Galois element `p`, from a source at scale
/// `2^40` into a narrower destination, with a key that covers the shaped
/// working width and one that does not, on a full and a replicated layout.
fn narrow_automorphisms<B, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<B>,
    p: i64,
    results: &mut Vec<(&'static str, Snapshot)>,
    apply: Automorphism<B, Rg>,
) where
    B: Backend<ZnxWord = i64> + CKKSPlaintextVecImpl<Rg> + CKKSRotateImpl<Rg>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B, Rg> + GLWEFill<B, Rg>,
{
    let b = params.base2k;
    let mut source_layout = replicated_at(params, 8 * b + 5);
    source_layout.glwe_layout.k = 130usize.into();
    source_layout.meta.scale = crate::Scale::Log(40);
    source_layout.meta.log_sparsity = 0;
    let shaped_layout = ShapingUnit::U16_40.layout(1).unwrap();
    let label = if p == -1 { "ckks_conjugate_into" } else { "ckks_rotate_into" };
    for slot_layout in [SlotLayout::Full, SlotLayout::Replicated(shaped_layout)] {
        source_layout.meta.slot_layout = slot_layout;
        let source = fixture_ciphertext::<B, Rg>(module, &source_layout, 91);
        let source_before = snapshot::<B, _, Rg>(&source);
        let mut destination_layout = source_layout;
        destination_layout.glwe_layout.k = 110usize.into();
        for key_bits in [110, 180] {
            let key_infos = key_layout_bits(params.n, b, key_bits, 10, params.rank, params.rank);
            let key = prepared_automorphism_key::<B, Rg>(module, &key_infos, p, 93);
            let mut destination = fixture_ciphertext::<B, Rg>(module, &destination_layout, 95);
            let before = snapshot::<B, _, Rg>(&destination);
            let bytes = CKKSRotateOps::<B, Rg>::ckks_rotate_tmp_bytes(module, &source, &key_infos);
            let result = with_scratch::<B, _>(bytes, |scratch| apply(module, &mut destination, &source, &key, scratch));
            if slot_layout != SlotLayout::Full && key_bits == 110 {
                let err = result.unwrap_err();
                let required = 110 + crate::shaping::ShapingProfile::new(&shaped_layout).shaping_depth(40);
                assert!(
                    matches!(err.composition(), Some(crate::CKKSCompositionError::MissingAutomorphismKey { op, k, .. }) if *op == label && *k as usize == required),
                    "{err:?}"
                );
                assert_eq!(snapshot::<B, _, Rg>(&destination), before);
            } else {
                result.unwrap();
                results.push((
                    if p == -1 { "narrow_conjugate" } else { "narrow_rotate" },
                    snapshot::<B, _, Rg>(&destination),
                ));
            }
            assert_eq!(snapshot::<B, _, Rg>(&source), source_before);
        }
    }
}

/// The conjugations of the standard ring.
fn conjugations<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + ReplicatedRingImpl<Standard> + CKKSConjugateImpl,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWEFill<B>,
{
    let b = params.base2k;
    let la = replicated_at(params, 8 * b + 5);
    let lr = replicated_at(params, 5 * b + 3);
    let a = fixture_ciphertext::<B, Standard>(module, &la, 71);
    let (_, atk_layout) = rank_keys(params, la.k().as_usize());
    let conj = prepared_automorphism_key::<B, Standard>(module, &atk_layout, -1, 63);
    let mut results = Vec::new();
    let mut out = fixture_ciphertext::<B, Standard>(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_conjugate_tmp_bytes(&a, &atk_layout), |s| {
        module.ckks_conjugate_into(&mut out, &a, &conj, s)
    })
    .unwrap();
    results.push(("conjugate_into", snapshot::<B, _, Standard>(&out)));
    narrow_automorphisms::<B, Standard>(params, module, -1, &mut results, |module, dst, src, key, scratch| {
        module.ckks_conjugate_into(dst, src, key, scratch)
    });
    results
}

/// Shaped operations of a replicated layout give the same digits and metadata
/// on both backends, on the ring `Rg` and at the suite's rank, each within its
/// queried arena: rounding of replicated and full layouts, of a source with its
/// canonical flag clear, to another radix and past the largest shaping pass;
/// the shaped encoding; products, rotations, the masked one and polynomial
/// evaluation; and, on the standard ring, the conjugations.
pub fn test_replicated_parity<BR, BT, F, Rg>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64>
        + ReplicatedRingImpl<Rg>
        + ReplicatedRingImpl<Standard>
        + CKKSConjugateImpl
        + CKKSEncodingImpl<F, Rg>,
    BT: Backend<ZnxWord = i64>
        + ReplicatedRingImpl<Rg>
        + ReplicatedRingImpl<Standard>
        + CKKSConjugateImpl
        + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
    Rg: CKKSSlotEmbedding,
    Module<BR>: CKKSAllOpsTmpBytes<BR>
        + GLWETensorKeyPreparedFactory<BR, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BR, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWEFill<BR, Rg>
        + GLWEFill<BR>
        + GLWEAdd<BR, Rg>
        + CKKSReplicatedOps<BR, Rg>
        + ReplicatedTransformModule<BR, F, Rg>
        + CnvPVecAlloc<BR>,
    Module<BT>: CKKSAllOpsTmpBytes<BT>
        + GLWETensorKeyPreparedFactory<BT, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BT, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWEFill<BT, Rg>
        + GLWEFill<BT>
        + GLWEAdd<BT, Rg>
        + CKKSReplicatedOps<BT, Rg>
        + ReplicatedTransformModule<BT, F, Rg>
        + CnvPVecAlloc<BT>,
{
    assert_eq!(reference.n(), tested.n());
    let standard = !conjugate_invariant::<Rg>();
    let run =
        |rounding: Vec<_>, shaped: Vec<_>, converted: Vec<_>, conjugations: Option<Vec<_>>| -> Vec<(&'static str, Snapshot)> {
            rounding
                .into_iter()
                .chain(shaped)
                .chain(converted)
                .chain(conjugations.into_iter().flatten())
                .collect()
        };
    let want = run(
        rounding_ops::<BR, F, Rg>(params, reference),
        shaped_ops::<BR, F, Rg>(params, reference),
        conversions::<BR, F, Rg>(params, reference),
        standard.then(|| conjugations::<BR>(params, reference)),
    );
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    let got = run(
        rounding_ops::<BT, F, Rg>(params, tested),
        shaped_ops::<BT, F, Rg>(params, tested),
        conversions::<BT, F, Rg>(params, tested),
        standard.then(|| conjugations::<BT>(params, tested)),
    );
    assert_eq!(want, got, "replicated parity differs");
}
