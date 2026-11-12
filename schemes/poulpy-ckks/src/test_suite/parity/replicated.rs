//! Shaped operations of a replicated slot layout: rounding and encoding. The
//! shaped encryption runs in [`test_encryption_parity`](super::test_encryption_parity).
use super::{
    arithmetic::layout,
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, fixture_operand, snapshot, with_scratch},
};
use crate::{
    CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingScalar, CKKSModuleInfos, CKKSRoundOps},
    layouts::CKKSPlaintextAlloc,
    oep::{CKKSEncodingImpl, CKKSRoundImpl},
    reference::encoding::CKKSSlotEmbedding,
    shaping::{ShapingProfile, ShapingRing, ShapingUnit},
    test_suite::CKKSTestParams,
};
use poulpy_core::{GLWEAdd, GLWEFill};
use poulpy_hal::layouts::{Backend, Module, Ring};

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

/// The roundings and the shaped encoding of the ring `Rg`.
fn replicated_ops<B, F, Rg>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
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
    let pass = ShapingProfile::with_ring(&unit, ShapingRing::of::<Rg>())
        .unwrap()
        .max_pass_bits();
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
    let im = if ShapingRing::of::<Rg>() == ShapingRing::ConjugateInvariant {
        0.0
    } else {
        -0.25
    };
    let (re, im) = ([value(0.625), value(-0.375)], [value(im), value(-im)]);
    with_scratch::<B, _>(module.ckks_encoding_host_tmp_bytes(slots), |s| {
        module.ckks_encode_slots_host_into(&mut pt, &re, &im, s)
    })
    .unwrap();
    results.push(("replicated encoding", snapshot::<B, _, Rg>(&pt)));
    results
}

/// Shaped rounding and encoding of a replicated layout give the same digits
/// and metadata on both backends, on the ring `Rg`, each within its queried
/// arena: replicated and full layouts, a source with its canonical flag clear,
/// a destination at another radix and a drop past the largest shaping pass.
pub fn test_replicated_parity<BR, BT, F, Rg>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSRoundImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    BT: Backend<ZnxWord = i64> + CKKSRoundImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
    Rg: CKKSSlotEmbedding,
    Module<BR>: GLWEFill<BR, Rg> + GLWEAdd<BR, Rg>,
    Module<BT>: GLWEFill<BT, Rg> + GLWEAdd<BT, Rg>,
{
    assert_eq!(reference.n(), tested.n());
    let want = replicated_ops::<BR, F, Rg>(params, reference);
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    assert_eq!(want, replicated_ops::<BT, F, Rg>(params, tested), "replicated parity differs");
}
