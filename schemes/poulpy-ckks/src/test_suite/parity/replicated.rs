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
use poulpy_hal::layouts::{Backend, Module, Ring, Standard};

/// The CKKS families of a ring that the shaped operations use.
pub trait ReplicatedRingImpl<Rg: Ring>: CKKSRoundImpl<Rg> {}

impl<B, Rg: Ring> ReplicatedRingImpl<Rg> for B where B: CKKSRoundImpl<Rg> {}

/// A replicated layout of period 8 holding 2 values, on `2kR = 32` slots.
fn replicated(params: CKKSTestParams, k: usize) -> CKKSLayout {
    let b = params.base2k;
    let mut out = layout(params, 1, k, b - 1, params.n.ilog2() as usize - 5, SlotsKind::Complex);
    out.meta.slot_layout = SlotLayout::Replicated(ShapingUnit::U16_20.layout(1, 8).unwrap());
    out
}

/// The shaped operations of the ring `Rg`.
fn replicated_ops<B, Rg: Ring>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + ReplicatedRingImpl<Rg>,
    Module<B>: GLWEFill<B, Rg>,
{
    let b = params.base2k;
    let la = replicated(params, 8 * b + 5);
    let a = fixture_ciphertext::<B, Rg>(module, &la, 71);
    let before = snapshot::<B, _, Rg>(&a);
    let mut results = Vec::new();

    let mut out = fixture_ciphertext::<B, Rg>(module, &replicated(params, la.k().as_usize() - b / 2), 99);
    with_scratch::<B, _>(CKKSRoundOps::<B, Rg>::ckks_round_tmp_bytes(module, &out, &a), |s| {
        module.ckks_round(&mut out, &a, s)
    })
    .unwrap();
    results.push(("round_into", snapshot::<B, _, Rg>(&out)));

    assert_eq!(before, snapshot::<B, _, Rg>(&a));
    results
}

/// Shaped rounding of a replicated layout gives the same digits and metadata
/// on both backends, on the standard ring.
pub fn test_replicated_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEFill<BR>,
    Module<BT>: GLWEFill<BT>,
{
    test_replicated_ring_parity::<BR, BT, F, Standard>(params, reference, tested)
}

/// [`test_replicated_parity`] on the ring `Rg`.
pub fn test_replicated_ring_parity<BR, BT, F, Rg: Ring>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + ReplicatedRingImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    BT: Backend<ZnxWord = i64> + ReplicatedRingImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEFill<BR, Rg>,
    Module<BT>: GLWEFill<BT, Rg>,
{
    assert_eq!(reference.n(), tested.n());
    let want = replicated_ops::<BR, Rg>(params, reference);
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    assert_eq!(
        want,
        replicated_ops::<BT, Rg>(params, tested),
        "replicated ring parity differs"
    );
}
