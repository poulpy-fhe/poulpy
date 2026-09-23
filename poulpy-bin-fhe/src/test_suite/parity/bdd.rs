//! Exact coefficient and metadata parity for selection and BDD operations.

use super::{GlweSnapshot, ParityBackend, fixture_ggsw, fixture_glwe, snapshot_ggsw, snapshot_glwe, with_scratch};
use crate::{api::*, bdd_arithmetic::*};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
use std::collections::HashMap;

/// Public operations and fixture preparation required for caller-selected parity.
pub trait BddParityModule<B: Backend>:
    Cmux<B>
    + Cswap<B>
    + GLWEBlindRotation<B>
    + GLWEBlindSelection<u8, B>
    + GLWEBlindRetrieval<B>
    + GGSWBlindRotation<u8, B>
    + ExecuteBDDCircuit<B>
    + GGSWPreparedFactory<B>
{
}
impl<B: Backend, M> BddParityModule<B> for M where
    M: Cmux<B>
        + Cswap<B>
        + GLWEBlindRotation<B>
        + GLWEBlindSelection<u8, B>
        + GLWEBlindRetrieval<B>
        + GGSWBlindRotation<u8, B>
        + ExecuteBDDCircuit<B>
        + GGSWPreparedFactory<B>
{
}

fn layouts<B: Backend>(module: &Module<B>) -> (GLWELayout, GGSWLayout) {
    (
        GLWELayout {
            n: (module.n() as u32).into(),
            base2k: 12u32.into(),
            k: 35u32.into(),
            rank: 1u32.into(),
        },
        GGSWLayout {
            n: (module.n() as u32).into(),
            base2k: 12u32.into(),
            dnum: 3u32.into(),
            dsize: 1u32.into(),
            k_aux: 12u32.into(),
            rank: 1u32.into(),
        },
    )
}

fn selector<B: ParityBackend>(module: &Module<B>) -> FheUintPrepared<B::OwnedBuf, u8, B>
where
    Module<B>: GGSWPreparedFactory<B>,
{
    let (_, key) = layouts(module);
    let mut result = FheUintPrepared::alloc_from_infos(module, &key);
    for (i, bit) in result.bits.iter_mut().enumerate() {
        let raw = fixture_ggsw(module, &key, 31 + i as u8);
        with_scratch::<B, _>(module.ggsw_prepare_tmp_bytes(&key), |s| module.ggsw_prepare(bit, &raw, s));
    }
    result
}

// The provider itself borrows prepared key storage, so requiring a static key
// type on any public selection/retrieval route fails to compile these fixtures.
struct BorrowedBits<'a, B: Backend>(&'a [GGSWPrepared<B::OwnedBuf, B>]);

impl<B: Backend> GetGGSWBit<B> for BorrowedBits<'_, B> {
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<B::OwnedBuf, B> {
        &self.0[bit]
    }
}

fn gates<B: ParityBackend>(module: &Module<B>, swap: bool) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (layout, _) = layouts(module);
    let selectors = selector(module);
    let bit = selectors.get_bit(0).to_backend_ref();
    let mut outputs = Vec::new();
    for variant in 0..if swap { 2 } else { 4 } {
        let mut other_layout = layout;
        if swap && variant == 1 {
            other_layout.base2k = 24u32.into();
        }
        let mut left = fixture_glwe(module, &other_layout, 2);
        // CMux's false branch may have more limbs than the true branch and destination.
        // Its immutable storage must not add an unadvertised scratch allocation.
        let mut false_layout = other_layout;
        if !swap && variant == 3 {
            false_layout.k = 179u32.into();
        }
        let mut right = fixture_glwe(module, &false_layout, 3);
        if swap {
            let bytes = module.cswap_tmp_bytes(&left, &right, &bit);
            with_scratch::<B, _>(bytes, |scratch| module.cswap(&mut left, &mut right, &bit, scratch));
            outputs.push(snapshot_glwe::<B, _>(&left));
            outputs.push(snapshot_glwe::<B, _>(&right));
        } else {
            let mut out = fixture_glwe(module, &layout, 2);
            let bytes = module.cmux_tmp_bytes(&out, &left, &bit);
            with_scratch::<B, _>(bytes, |scratch| match variant {
                0 | 3 => module.cmux(&mut out, &left, &right, &bit, scratch),
                1 => module.cmux_assign(&mut out, &right, &bit, scratch),
                _ => module.cmux_assign_neg(&mut out, &right, &bit, scratch),
            });
            outputs.push(snapshot_glwe::<B, _>(&out));
        }
    }
    outputs
}

/// Checks all CMux method variants with exact advertised scratch.
pub fn test_cmux_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(gates(reference, false), gates(tested, false));
}
/// Checks conditional swap, including a selector radix different from the operands.
pub fn test_cswap_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(gates(reference, true), gates(tested, true));
}

fn rotations<B: ParityBackend>(module: &Module<B>) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (layout, key) = layouts(module);
    let selectors = selector(module);
    let mut outputs = Vec::new();
    for sign in [false, true] {
        for mask in [0, 1, 3] {
            let mut out = fixture_glwe(module, &layout, 7);
            with_scratch::<B, _>(module.glwe_blind_rotation_assign_tmp_bytes(&out, &key), |s| {
                module.glwe_blind_rotation_assign(&mut out, &selectors, sign, 1, mask, 1, s)
            });
            outputs.push(snapshot_glwe::<B, _>(&out));
            for input_layout in [
                layout,
                GLWELayout {
                    k: 60usize.into(),
                    ..layout
                },
                GLWELayout {
                    k: 60usize.into(),
                    base2k: 10usize.into(),
                    ..layout
                },
            ] {
                let input = fixture_glwe(module, &input_layout, 7);
                let mut out = fixture_glwe(module, &layout, 7);
                let bytes = module.glwe_blind_rotation_tmp_bytes(&out, &input, &key);
                with_scratch::<B, _>(bytes, |s| {
                    module.glwe_blind_rotation(&mut out, &input, &selectors, sign, 1, mask, 1, s)
                });
                outputs.push(snapshot_glwe::<B, _>(&out));
            }
        }
    }
    // Preserve larger allocation capacities after reducing active precision.
    let capacity_layout = GLWELayout {
        k: 60usize.into(),
        ..layout
    };
    let mut input = fixture_glwe(module, &capacity_layout, 7);
    input.set_k(layout.k);
    let mut out = fixture_glwe(module, &capacity_layout, 7);
    out.set_k(layout.k);
    let bytes = module.glwe_blind_rotation_tmp_bytes(&out, &input, &key);
    with_scratch::<B, _>(bytes, |s| {
        module.glwe_blind_rotation(&mut out, &input, &selectors, false, 1, 3, 1, s)
    });
    outputs.push(snapshot_glwe::<B, _>(&out));
    let bytes = module.glwe_blind_rotation_assign_tmp_bytes(&out, &key);
    with_scratch::<B, _>(bytes, |s| {
        module.glwe_blind_rotation_assign(&mut out, &selectors, true, 1, 3, 1, s)
    });
    outputs.push(snapshot_glwe::<B, _>(&out));
    outputs
}
/// Checks both rotation directions, offsets, empty bit ranges, and assignment variants.
pub fn test_glwe_blind_rotation_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(rotations(reference), rotations(tested));
}

fn selection_fixture<B: ParityBackend>(
    module: &Module<B>,
    allocation: &GLWELayout,
    precision: TorusPrecision,
    seed: u8,
) -> GLWE<B::OwnedBuf, i64> {
    let mut value = fixture_glwe(module, allocation, seed);
    value.set_k(precision);
    let mut host = value.to_host_owned::<B>();
    super::canonicalize(&mut host);
    poulpy_core::TransferInto::transfer_into(&host, &mut value);
    value
}

fn selection<B: ParityBackend>(module: &Module<B>) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let selectors = selector(module);
    let expected = selection_with_key(module, &selectors);
    assert_eq!(expected, selection_with_key(module, &BorrowedBits(&selectors.bits)));
    expected
}

fn selection_with_key<B: ParityBackend, K: GetGGSWBit<B>>(module: &Module<B>, selectors: &K) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (layout, key) = layouts(module);
    let mut outputs = Vec::new();
    for varied in [false, true] {
        for present in [0b1111, 0b0101, 0b1010, 0] {
            let mut values: Vec<_> = (0..4)
                .map(|i| {
                    let mut input_layout = layout;
                    if varied {
                        input_layout.k = [23u32, 47, 59, 71][i].into();
                    }
                    let precision = if varied { [23u32, 35, 47, 59][i].into() } else { layout.k };
                    selection_fixture(module, &input_layout, precision, 10 + i as u8)
                })
                .collect();
            let mut output_allocation = layout;
            if varied {
                output_allocation.k = 84u32.into();
            }
            let mut out = selection_fixture(module, &output_allocation, layout.k, 99);
            let bytes = {
                let inputs: Vec<_> = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| present & (1 << i) != 0)
                    .map(|(_, input)| input)
                    .collect();
                module.glwe_blind_selection_tmp_bytes(&out, &inputs, &key)
            };
            let map: HashMap<_, _> = values
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| present & (1 << i) != 0)
                .collect();
            with_scratch::<B, _>(bytes, |s| module.glwe_blind_selection(&mut out, map, selectors, 1, 2, s));
            outputs.push(snapshot_glwe::<B, _>(&out));
        }
    }
    outputs
}
/// Checks complete, sparse, and empty selections, including differing input
/// precisions and capacities and both orientations of implicit zero branches.
/// Borrowed key providers must match the owned provider on each backend.
pub fn test_glwe_blind_selection_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(selection(reference), selection(tested));
}

fn retrieval<B: ParityBackend>(module: &Module<B>) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let selectors = selector(module);
    let expected = retrieval_with_key(module, &selectors);
    assert_eq!(expected, retrieval_with_key(module, &BorrowedBits(&selectors.bits)));
    expected
}

fn retrieval_with_key<B: ParityBackend, K: GetGGSWBit<B>>(module: &Module<B>, selectors: &K) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (layout, key) = layouts(module);
    let mut outputs = Vec::new();
    let mut values: Vec<_> = (0..4).map(|i| fixture_glwe(module, &layout, 10 + i)).collect();
    let bytes = module.glwe_blind_retrieval_tmp_bytes(&layout, &key);
    for reverse in [false, true, false] {
        with_scratch::<B, _>(bytes, |s| match reverse {
            false => module.glwe_blind_retrieval_statefull(&mut values, selectors, 1, 2, s),
            true => module.glwe_blind_retrieval_statefull_rev(&mut values, selectors, 1, 2, s),
        });
        outputs.extend(values.iter().map(snapshot_glwe::<B, _>));
    }
    outputs
}
/// Checks forward/reverse retrieval and repeated use of the same state.
/// Borrowed key providers must match the owned provider on each backend.
pub fn test_glwe_blind_retrieval_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(retrieval(reference), retrieval(tested));
}

struct TinyCircuit;
impl GetBitCircuitInfo for TinyCircuit {
    fn input_size(&self) -> usize {
        2
    }
    fn output_size(&self) -> usize {
        2
    }
    fn get_circuit(&self, _: usize) -> (&[Node], usize) {
        (&[Node::Cmux(0, 1, 0), Node::Copy, Node::Cmux(1, 0, 1), Node::None], 2)
    }
}
fn evaluation<B: ParityBackend>(module: &Module<B>) -> Vec<GlweSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (layout, key) = layouts(module);
    let inputs = selector(module);
    let mut outputs = Vec::new();
    for threads in [1, 2, 8] {
        let mut out: Vec<_> = (0..3).map(|i| fixture_glwe(module, &layout, 50 + i)).collect();
        let per_worker = module.execute_bdd_circuit_tmp_bytes_for(&layout, &TinyCircuit, &key);
        let workers = poulpy_hal::execution::worker_count::<B::TaskExecutor>(threads, 2);
        let bytes = workers * poulpy_hal::execution::worker_scratch_bytes::<B>(per_worker);
        with_scratch::<B, _>(bytes, |s| {
            module.execute_bdd_circuit_multi_thread(threads, &mut out, &inputs, &TinyCircuit, s)
        });
        outputs.extend(out.iter().map(snapshot_glwe::<B, _>));
    }
    outputs
}
/// Checks BDD copy/CMux levels, zero output tails and several worker requests.
pub fn test_execute_bdd_circuit_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(evaluation(reference), evaluation(tested));
}

fn matrix_rotations<B: ParityBackend>(module: &Module<B>) -> Vec<super::GgswSnapshot>
where
    Module<B>: BddParityModule<B>,
{
    let (_, key) = layouts(module);
    let inputs = selector(module);
    let mut outputs = Vec::new();
    let mut scalar = vec![0u8; module.n() * size_of::<i64>()];
    scalar[..8].copy_from_slice(&1i64.to_ne_bytes());
    let scalar = ScalarZnx::from_data(B::from_host_bytes(&scalar), module.n(), 1);
    for variant in 0..3 {
        let input = fixture_ggsw(module, &key, 61);
        let mut out = fixture_ggsw(module, &key, 61);
        let bytes = if variant == 2 {
            module.scalar_to_ggsw_blind_rotation_tmp_bytes(&key, &key)
        } else if variant == 1 {
            module.ggsw_blind_rotation_assign_tmp_bytes(&out, &key)
        } else {
            module.ggsw_to_ggsw_blind_rotation_tmp_bytes(&out, &input, &key)
        };
        with_scratch::<B, _>(bytes, |s| match variant {
            0 => module.ggsw_blind_rotation(&mut out, &input, &inputs, true, 1, 2, 0, s),
            1 => module.ggsw_blind_rotation_assign(&mut out, &inputs, false, 1, 2, 0, s),
            _ => module.scalar_to_ggsw_blind_rotation(&mut out, &scalar, &inputs, true, 1, 2, 0, s),
        });
        outputs.push(snapshot_ggsw::<B, _>(&out));
    }
    for input_layout in [
        GGSWLayout {
            k_aux: 24usize.into(),
            ..key
        },
        GGSWLayout {
            base2k: 10usize.into(),
            k_aux: 30usize.into(),
            ..key
        },
    ] {
        let input = fixture_ggsw(module, &input_layout, 61);
        let mut out = fixture_ggsw(module, &key, 61);
        let bytes = module.ggsw_to_ggsw_blind_rotation_tmp_bytes(&out, &input, &key);
        with_scratch::<B, _>(bytes, |s| {
            module.ggsw_blind_rotation(&mut out, &input, &inputs, true, 1, 2, 0, s)
        });
        outputs.push(snapshot_ggsw::<B, _>(&out));
    }
    outputs
}
/// Checks both GGSW rotation variants and construction from a scalar test vector.
pub fn test_ggsw_blind_rotation_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BddParityModule<BR>,
    Module<BT>: BddParityModule<BT>,
{
    assert_eq!(matrix_rotations(reference), matrix_rotations(tested));
}
