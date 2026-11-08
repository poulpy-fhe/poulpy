//! Shaped operations of a replicated slot layout: rounding.
use super::{
    arithmetic::layout,
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, snapshot, with_scratch},
};
use crate::{
    CKKSLayout, SlotLayout, SlotsKind,
    api::{CKKSEncodingScalar, CKKSRoundOps},
    oep::{CKKSEncodingImpl, CKKSImpl, CKKSRoundImpl},
    shaping::ShapingUnit,
    test_suite::CKKSTestParams,
};
use poulpy_core::{GLWEFill, layouts::LWEInfos};
use poulpy_hal::layouts::{Backend, Module};

/// The CKKS families of both rings that the shaped operations use.
pub trait ReplicatedRingImpl: CKKSRoundImpl {}

impl<B> ReplicatedRingImpl for B where B: CKKSRoundImpl {}

/// A replicated layout of period 8 holding 2 values, on `2kR = 32` slots.
fn replicated(params: CKKSTestParams, k: usize) -> CKKSLayout {
    let b = params.base2k;
    let mut out = layout(params, 1, k, b - 1, params.n.ilog2() as usize - 5, SlotsKind::Complex);
    out.meta.slot_layout = SlotLayout::Replicated(ShapingUnit::U16_20.layout(1, 8).unwrap());
    out
}

/// The shaped operations of both rings.
fn replicated_ops<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + ReplicatedRingImpl,
    Module<B>: GLWEFill<B>,
{
    let b = params.base2k;
    let la = replicated(params, 8 * b + 5);
    let a = fixture_ciphertext(module, &la, 71);
    let before = snapshot::<B, _>(&a);
    let mut results = Vec::new();

    let mut out = fixture_ciphertext(module, &replicated(params, la.k().as_usize() - b / 2), 99);
    with_scratch::<B, _>(module.ckks_round_tmp_bytes(&out, &a), |s| module.ckks_round(&mut out, &a, s)).unwrap();
    results.push(("round_into", snapshot::<B, _>(&out)));

    assert_eq!(before, snapshot::<B, _>(&a));
    results
}

/// Shaped rounding of a replicated layout gives the same digits and metadata
/// on both backends.
pub fn test_replicated_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEFill<BR>,
    Module<BT>: GLWEFill<BT>,
{
    test_replicated_ring_parity::<BR, BT, F>(params, reference, tested)
}

/// [`test_replicated_parity`] on either ring.
pub fn test_replicated_ring_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + ReplicatedRingImpl + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + ReplicatedRingImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEFill<BR>,
    Module<BT>: GLWEFill<BT>,
{
    assert_eq!(reference.n(), tested.n());
    let want = replicated_ops::<BR>(params, reference);
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    assert_eq!(want, replicated_ops::<BT>(params, tested), "replicated ring parity differs");
}
