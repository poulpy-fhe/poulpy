//! Shaped operations of a replicated slot layout: rounding, encoding, products,
//! automorphisms and polynomial evaluation with masked constants, the conversions
//! and the ring maps. The shaped encryption runs in [`test_encryption_parity`](super::test_encryption_parity).
use super::{
    arithmetic::layout,
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, fixture_operand, fixture_plaintext, snapshot, with_scratch},
    keys::{key_layout, key_layout_bits, prepared_automorphism_key, prepared_tensor_key},
};
use crate::{
    CKKSInfos, CKKSLayout, CKKSModuleInfos, CoeffsMeta, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSCIRingMapOps, CKKSConjugateOps, CKKSEncodingHostOps, CKKSEncodingScalar, CKKSMulOps,
        CKKSPolynomialEvaluationOps, CKKSReplicatedOps, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSPlaintextAlloc, ReplicatedConversionPrepared, eval_mod::encode_bsgs_with,
        masked::encode_masked_constants, replicated::ReplicatedConversionModule,
    },
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSDotProductImpl, CKKSEncodingImpl, CKKSMulImpl, CKKSPlaintextVecImpl,
        CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRotateImpl, CKKSRoundImpl, CKKSSubImpl,
    },
    polynomial::{Basis, Polynomial, SplitStrategy},
    reference::encoding::CKKSSlotEmbedding,
    shaping::{ShapingProfile, ShapingUnit, conjugate_invariant},
    test_suite::{CKKSTestParams, replicated::round_trip_conversion},
};
use poulpy_core::{
    GLWEAdd, GLWEAutomorphismKeyPreparedFactory, GLWEFill, GLWETensorKeyPreparedFactory,
    api::GLWECIKeyPreparedFactory,
    layouts::{GGLWELayout, GLWEAutomorphismKeyPrepared, GLWELayout, GLWESwitchingKeyAlloc, LWEInfos},
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, ScratchOwned, Standard},
    source::Source,
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

/// The CKKS, replicated and diagonal scales of the conversions of [`run_replicated_conversions`].
const CONVERSION_SCALES: (usize, usize, usize) = (40, 20, 30);

/// A conversion of [`run_replicated_conversions`] with what running it takes: its automorphism key layout at the
/// suite's rank, fixture keys of its Galois elements and the `k` of its round trip.
struct ConversionSetup<B: Backend, Rg: Ring> {
    conv: ReplicatedConversionPrepared<B, Rg>,
    atk_layout: GGLWELayout,
    keys: HashMap<i64, GLWEAutomorphismKeyPrepared<B::OwnedBuf, B, Rg>>,
    k: usize,
}

/// The conversion of `k = 4` values onto `U8_40` (`R = 4`) at [`CONVERSION_SCALES`], dense or factored into
/// `placement_factors`.
fn conversion_setup<B, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<B>,
    placement_factors: Option<usize>,
) -> ConversionSetup<B, Rg>
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<B>: ReplicatedConversionModule<B, F, Rg> + GLWEAutomorphismKeyPreparedFactory<B, Rg> + GLWEFill<B, Rg>,
{
    let b = params.base2k;
    let unit = ShapingUnit::U8_40.layout(2).unwrap();
    let (conv, (.., extra_bits), k) =
        round_trip_conversion::<B, F, Rg>(module, (params.n, unit, b.into()), CONVERSION_SCALES, placement_factors).unwrap();
    let (_, atk_layout) = rank_keys(params, k + extra_bits + conv.trace_levels + b);
    let key = |p: i64| prepared_automorphism_key::<B, Rg>(module, &atk_layout, p, (p as u8).wrapping_add(31));
    let keys = conv.galois_elements().into_iter().map(|p| (p, key(p))).collect();
    ConversionSetup {
        conv,
        atk_layout,
        keys,
        k,
    }
}

/// `ckks_to_replicated` and back of full-layout fixtures of `k = 4` values onto `U8_40` (`R = 4`), complex, complex with
/// the canonical flag clear and real, dense and with two placement factors, into wide and narrower destinations within
/// their queried arenas, keys at the suite's rank, each source left unchanged.
pub fn run_replicated_conversions<B, F, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSReplicatedOps<B, Rg>
        + ReplicatedConversionModule<B, F, Rg>
        + GLWEAutomorphismKeyPreparedFactory<B, Rg>
        + GLWEFill<B, Rg>
        + GLWEAdd<B, Rg>,
{
    let (log_delta_ckks, log_delta, _) = CONVERSION_SCALES;
    let full = |k| layout(params, params.rank, k, log_delta_ckks, 0, SlotsKind::Complex);
    let mut results = Vec::new();
    for (label, placement_factors) in [("dense", None), ("factored", Some(2))] {
        let ConversionSetup {
            conv,
            atk_layout,
            keys,
            k,
        } = conversion_setup::<B, F, Rg>(params, module, placement_factors);
        let convert = |res: &mut CKKSCiphertextOwned<B, Rg>, a: &CKKSCiphertextOwned<B, Rg>, to: bool| {
            let before = snapshot::<B, _, Rg>(a);
            with_scratch::<B, _>(
                module.ckks_replicated_conversion_tmp_bytes(&*res, a, &conv, &atk_layout),
                |s| {
                    if to {
                        module.ckks_to_replicated(res, a, &conv, &keys, s)
                    } else {
                        module.ckks_from_replicated(res, a, &conv, &keys, s)
                    }
                },
            )
            .unwrap();
            assert_eq!(snapshot::<B, _, Rg>(a), before, "a conversion changed its source");
            (label, snapshot::<B, _, Rg>(&*res))
        };
        // The results land wide, and narrower off the limb grid through a modulus drop.
        let to_k = k - log_delta_ckks - conv.to.consumed + log_delta;
        let from_log_delta = log_delta + conv.extra_bits + conv.trace_levels;
        for (slots, lazy) in [
            (SlotsKind::Complex, false),
            (SlotsKind::Complex, true),
            (SlotsKind::Real, false),
        ] {
            let mut source = full(k);
            source.meta.slots = slots;
            let a = fixture_operand::<B, Rg>(module, &source, 81, lazy);
            let mut onto = None;
            for res_k in [k, to_k - 7] {
                let res_layout = ct_layout(params, res_k, log_delta, SlotLayout::Replicated(conv.layout));
                let mut res = fixture_ciphertext::<B, Rg>(module, &res_layout, 83);
                results.push(convert(&mut res, &a, true));
                onto.get_or_insert(res);
            }
            let onto = onto.unwrap();
            for res_k in [k, from_log_delta + 7] {
                let mut back = fixture_ciphertext::<B, Rg>(module, &full(res_k), 85);
                results.push(convert(&mut back, &onto, false));
            }
        }
    }
    results
}

/// A recordable run of both conversions on `B`, as [`run_replicated_conversions`] builds them: the dense conversion of
/// `k = 4` values onto `U8_40` (`R = 4`), fixture keys and a full-layout fixture source. Each call converts the source
/// onto the replicated layout and back on one arena allocated once, without waiting for the backend.
pub fn replicated_conversions_run<B, F, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> impl FnMut() + '_
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSReplicatedOps<B, Rg>
        + ReplicatedConversionModule<B, F, Rg>
        + GLWEAutomorphismKeyPreparedFactory<B, Rg>
        + GLWEFill<B, Rg>
        + GLWEAdd<B, Rg>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let (log_delta_ckks, log_delta, _) = CONVERSION_SCALES;
    let ConversionSetup {
        conv,
        atk_layout,
        keys,
        k,
    } = conversion_setup::<B, F, Rg>(params, module, None);
    let full = layout(params, params.rank, k, log_delta_ckks, 0, SlotsKind::Complex);
    let source = fixture_operand::<B, Rg>(module, &full, 81, false);
    let onto_layout = ct_layout(params, k, log_delta, SlotLayout::Replicated(conv.layout));
    let mut onto = fixture_ciphertext::<B, Rg>(module, &onto_layout, 83);
    let mut back = fixture_ciphertext::<B, Rg>(module, &full, 85);
    let bytes = module
        .ckks_replicated_conversion_tmp_bytes(&onto, &source, &conv, &atk_layout)
        .max(module.ckks_replicated_conversion_tmp_bytes(&back, &onto, &conv, &atk_layout));
    let mut scratch = ScratchOwned::<B>::alloc(bytes);
    move || {
        module
            .ckks_to_replicated(&mut onto, &source, &conv, &keys, &mut scratch.borrow())
            .unwrap();
        module
            .ckks_from_replicated(&mut back, &onto, &conv, &keys, &mut scratch.borrow())
            .unwrap();
    }
}

/// The refusals of `ckks_to_replicated` and `ckks_from_replicated` of the dense conversion of
/// [`run_replicated_conversions`] into a destination of another rank, each destination untouched.
pub fn run_refused_replicated_conversions<B, F, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> [crate::CKKSError; 2]
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSReplicatedOps<B, Rg>
        + ReplicatedConversionModule<B, F, Rg>
        + GLWEAutomorphismKeyPreparedFactory<B, Rg>
        + GLWEFill<B, Rg>,
{
    let (log_delta_ckks, log_delta, _) = CONVERSION_SCALES;
    let ConversionSetup { conv, keys, k, .. } = conversion_setup::<B, F, Rg>(params, module, None);
    let full = layout(params, params.rank, k, log_delta_ckks, 0, SlotsKind::Complex);
    let replicated = ct_layout(params, k, log_delta, SlotLayout::Replicated(conv.layout));
    [true, false].map(|to| {
        let (src, mut dst) = if to { (full, replicated) } else { (replicated, full) };
        dst.glwe_layout.rank = (params.rank + 1).into();
        let a = fixture_ciphertext::<B, Rg>(module, &src, 81);
        let mut res = fixture_ciphertext::<B, Rg>(module, &dst, 83);
        let before = snapshot::<B, _, Rg>(&res);
        let error = with_scratch::<B, _>(0, |s| {
            if to {
                module.ckks_to_replicated(&mut res, &a, &conv, &keys, s)
            } else {
                module.ckks_from_replicated(&mut res, &a, &conv, &keys, s)
            }
        })
        .expect_err("a conversion into another rank was accepted");
        assert_eq!(
            snapshot::<B, _, Rg>(&res),
            before,
            "a refused conversion wrote its destination"
        );
        error
    })
}

/// The replicated layout of the suite: period 8 holding 2 values, on `2kR = 32` slots.
fn replicated() -> SlotLayout {
    SlotLayout::Replicated(ShapingUnit::U16_20.layout(1).unwrap())
}

/// A ciphertext layout of the suite's rank at `k` and scale `2^delta`, in `slot_layout`.
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

/// A ciphertext layout of the suite's rank at `k`, in the replicated layout.
fn replicated_at(params: CKKSTestParams, k: usize) -> CKKSLayout {
    ct_layout(params, k, params.base2k - 1, replicated())
}

/// Tensor and automorphism key layouts of the suite's rank at `k`.
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
    with_scratch::<B, _>(CKKSMulOps::<B, Rg>::ckks_mul_pt_tmp_bytes(module, &out, &a, &pt), |s| {
        module.ckks_mul_pt_into(&mut out, &a, &pt, s)
    })
    .unwrap();
    results.push(("mul_pt_into", snapshot::<B, _, Rg>(&out)));
    // A scalar of degree 1 on the full layout, and a masked constant of degree `2R` (`R` on the conjugate-invariant
    // ring) on the replicated one.
    let crate::SlotLayout::Replicated(layout) = lp.meta.slot_layout else {
        unreachable!("a replicated plaintext layout")
    };
    let period = (1usize << layout.log_r()) * if crate::shaping::conjugate_invariant::<Rg>() { 1 } else { 2 };
    for (label, n, slot_layout) in [
        ("mul_pt_into scalar", 1, crate::SlotLayout::Full),
        ("mul_pt_into masked constant", period, lp.meta.slot_layout),
    ] {
        let mut cl = lp;
        cl.glwe_layout.n = n.into();
        cl.meta.slot_layout = slot_layout;
        let pt = fixture_plaintext::<B, Rg>(module, &cl, 55);
        let mut out = fixture_ciphertext::<B, Rg>(module, &lr, 99);
        with_scratch::<B, _>(CKKSMulOps::<B, Rg>::ckks_mul_pt_tmp_bytes(module, &out, &a, &pt), |s| {
            module.ckks_mul_pt_into(&mut out, &a, &pt, s)
        })
        .unwrap();
        results.push((label, snapshot::<B, _, Rg>(&out)));
    }

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
        let masked_meta = CoeffsMeta {
            meta: GLWEPlaintextMeta {
                slot_layout: SlotLayout::Replicated(rl),
                ..coeff_meta.meta
            },
            ..coeff_meta
        };
        let bsgs = with_scratch::<B, _>(encoding, |s| {
            encode_bsgs_with(&polynomial, SplitStrategy::MinDepth, false, &mut |values: &[F]| {
                encode_masked_constants::<B, F, Rg>(module, b.into(), masked_meta, values, s)
            })
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
                    matches!(err.composition(), Some(crate::CKKSCompositionError::AutomorphismKeyTooNarrow { op, k, max_input_k: 110, .. }) if *op == label && *k as usize == required),
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

/// `ckks_ci_embed` of a replicated conjugate-invariant fixture of half the degree and `ckks_ci_trace` of a replicated
/// standard one, each at the shaping depth within its queried arena, on fixture keys covering that depth, from a
/// canonical and a flag-clear source and into a destination at the result's `k` and a limb wider, each source left
/// unchanged.
fn ring_maps<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: CKKSCIRingMapOps<B>
        + GLWECIKeyPreparedFactory<B>
        + GLWEFill<B>
        + GLWEFill<B, ConjugateInvariant>
        + GLWEAdd<B>
        + GLWEAdd<B, ConjugateInvariant>,
{
    let b = params.base2k;
    let mut ci_layout = replicated_at(
        CKKSTestParams {
            n: params.n / 2,
            ..params
        },
        4 * b + 3,
    );
    ci_layout.meta.slots = SlotsKind::Real;
    let SlotLayout::Replicated(unit) = replicated() else {
        unreachable!()
    };
    let k = ci_layout.k().as_usize();
    let depth = ShapingProfile::new(&unit).shaping_depth(ci_layout.log_delta());
    let key_infos = key_layout(params.n, b, k + depth, 1, params.rank, params.rank);
    let switching = |seed: u8| {
        let mut key = module.glwe_switching_key_alloc_from_infos(&key_infos);
        fill_by_digit(module, &mut key, 1, &mut Source::new([seed; 32]));
        key
    };
    let mut embed = module.glwe_ci_embed_key_prepared_alloc_from_infos(&key_infos);
    let mut trace = module.glwe_ci_trace_key_prepared_alloc_from_infos(&key_infos);
    with_scratch::<B, _>(module.glwe_ci_key_prepare_tmp_bytes(&key_infos), |s| {
        module.glwe_ci_key_prepare(&mut embed, &switching(101), s);
        module.glwe_ci_key_prepare(&mut trace, &switching(103), s);
    });
    let mut standard_layout = replicated_at(params, k);
    standard_layout.meta.slots = SlotsKind::Real;
    let wider = |layout: &CKKSLayout, extra: usize| CKKSLayout {
        glwe_layout: GLWELayout {
            k: (layout.k().as_usize() + extra).into(),
            ..layout.glwe_layout
        },
        ..*layout
    };
    let mut results = Vec::new();
    for (labels, lazy, extra) in [
        (["ckks_ci_embed", "ckks_ci_trace"], false, 0),
        (["ckks_ci_embed, lazy", "ckks_ci_trace, lazy"], true, 0),
        (["ckks_ci_embed, wider", "ckks_ci_trace, wider"], false, b),
    ] {
        let src = fixture_operand::<B, ConjugateInvariant>(module, &ci_layout, 71, lazy);
        let src_before = snapshot::<B, _, ConjugateInvariant>(&src);
        let mut embedded = fixture_ciphertext::<B, Standard>(module, &wider(&replicated_at(params, k), extra), 83);
        with_scratch::<B, _>(module.ckks_ci_embed_tmp_bytes(&embedded, &src, &key_infos), |s| {
            module.ckks_ci_embed(&mut embedded, &src, &embed, s)
        })
        .unwrap();
        assert_eq!(
            snapshot::<B, _, ConjugateInvariant>(&src),
            src_before,
            "ckks_ci_embed changed its source"
        );
        let standard = fixture_operand::<B, Standard>(module, &standard_layout, 87, lazy);
        let standard_before = snapshot::<B, _, Standard>(&standard);
        let mut back = fixture_ciphertext::<B, ConjugateInvariant>(module, &wider(&ci_layout, extra), 85);
        with_scratch::<B, _>(module.ckks_ci_trace_tmp_bytes(&back, &standard, &key_infos), |s| {
            module.ckks_ci_trace(&mut back, &standard, &trace, s)
        })
        .unwrap();
        assert_eq!(
            snapshot::<B, _, Standard>(&standard),
            standard_before,
            "ckks_ci_trace changed its source"
        );
        results.push((labels[0], snapshot::<B, _, Standard>(&embedded)));
        results.push((labels[1], snapshot::<B, _, ConjugateInvariant>(&back)));
    }
    results
}

/// Shaped operations of a replicated layout give the same digits and metadata
/// on both backends, on the ring `Rg` and at the suite's rank, each within its
/// queried arena: rounding of replicated and full layouts, of a source with its
/// canonical flag clear, to another radix and past the largest shaping pass;
/// the shaped encoding; products, rotations, the masked one and polynomial
/// evaluation; the conversions; and, on the standard ring, the conjugations, on
/// the conjugate-invariant ring, the ring maps of a replicated source.
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
        + GLWEAdd<BR, Standard>
        + GLWEAdd<BR, ConjugateInvariant>
        + CKKSReplicatedOps<BR, Rg>
        + ReplicatedConversionModule<BR, F, Rg>
        + CKKSCIRingMapOps<BR>
        + GLWECIKeyPreparedFactory<BR>
        + GLWEFill<BR, ConjugateInvariant>,
    Module<BT>: CKKSAllOpsTmpBytes<BT>
        + GLWETensorKeyPreparedFactory<BT, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BT, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWEFill<BT, Rg>
        + GLWEFill<BT>
        + GLWEAdd<BT, Rg>
        + GLWEAdd<BT, Standard>
        + GLWEAdd<BT, ConjugateInvariant>
        + CKKSReplicatedOps<BT, Rg>
        + ReplicatedConversionModule<BT, F, Rg>
        + CKKSCIRingMapOps<BT>
        + GLWECIKeyPreparedFactory<BT>
        + GLWEFill<BT, ConjugateInvariant>,
{
    assert_eq!(reference.n(), tested.n());
    let standard = !conjugate_invariant::<Rg>();
    let mut want = rounding_ops::<BR, F, Rg>(params, reference);
    want.extend(shaped_ops::<BR, F, Rg>(params, reference));
    want.extend(run_replicated_conversions::<BR, F, Rg>(params, reference));
    want.extend(if standard {
        conjugations::<BR>(params, reference)
    } else {
        ring_maps::<BR>(params, reference)
    });
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    let mut got = rounding_ops::<BT, F, Rg>(params, tested);
    got.extend(shaped_ops::<BT, F, Rg>(params, tested));
    got.extend(run_replicated_conversions::<BT, F, Rg>(params, tested));
    got.extend(if standard {
        conjugations::<BT>(params, tested)
    } else {
        ring_maps::<BT>(params, tested)
    });
    assert_eq!(want, got, "replicated parity differs");
}
