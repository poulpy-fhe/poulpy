//! Trace, packing, relinearization and tensor-secret parity.
use super::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe, unnormalized_twin};
use crate::{
    Distribution, GLWEMaskFill, GLWEMulRelinearizeSum, GLWENormalize, GLWEPacking, GLWEPreparedTensoring, GLWETensorDecrypt,
    GLWETensoring, GLWETrace, GetDistribution,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretTensorFactory, GLWESecretTensorPreparedFactory,
        GLWETensorKeyPreparedFactory, TransferInto,
    },
    glwe_prepare_left, glwe_prepare_left_tmp_bytes, glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        Base2K, Degree, Dnum, GGLWEInfos, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWESecretLayout,
        GLWETensorKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision,
    },
    test_suite::{decryption::panic_message, keys::fill_by_digit, noise::assert_refuses_uncovered_input},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, Ring, ScratchOwned, Standard, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};
use std::collections::HashMap;

/// Covers complete, partial, summed, reversed and empty traces on either ring, the empty trace without keys, and
/// the refusal of levels outside the ring's trace before any write, by both entry points and the key query.
pub fn test_glwe_trace_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETrace<BR> + GLWEAutomorphismKeyPreparedFactory<BR>,
    Module<BT>: GLWETrace<BT> + GLWEAutomorphismKeyPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let b = params.base2k;
    let mut source = Source::new([131; 32]);
    let factor = <BR::Ring as Ring>::CYCLOTOMIC_ORDER_FACTOR;
    // The conjugate-invariant ring is fixed by the conjugation: its trace starts at level 1.
    let first = usize::from(factor != Standard::CYCLOTOMIC_ORDER_FACTOR);
    let end = (factor as usize * params.n).ilog2() as usize - 1;
    let n = Degree(params.n as u32);
    for reverse in [false, true] {
        assert_eq!(
            r.glwe_trace_galois_elements(n, first..end, reverse),
            t.glwe_trace_galois_elements(n, first..end, reverse)
        );
    }
    let mut refused = vec![
        (end..end + 1, "trace: levels outside the ring's trace"),
        (end..first, "trace: levels outside the ring's trace"),
    ];
    if first > 0 {
        refused.push((0..first, "trace: the conjugate-invariant ring has no conjugation level"));
    }
    for &rank in &shapes.ranks {
        let g = GLWELayout {
            n: Degree(params.n as u32),
            base2k: Base2K(b as u32),
            k: TorusPrecision((2 * b + 1) as u32),
            rank: Rank(rank as u32),
        };
        let k = GLWEAutomorphismKeyLayout {
            n: g.n,
            base2k: Base2K((b - 1) as u32),
            dnum: Dnum((2 * b + 1).div_ceil(b - 1) as u32),
            k_digit: TorusPrecision((b - 1) as u32),
            k_aux: TorusPrecision(b as u32 + 1),
            rank: g.rank,
        };
        let mut keys_r = HashMap::new();
        let mut keys_t = HashMap::new();
        // The forward rotations, then the inverses the reverse traces add.
        for p in r
            .glwe_trace_galois_elements(n, first..end, false)
            .into_iter()
            .chain(r.glwe_trace_galois_elements(n, first..end, true))
        {
            if keys_r.contains_key(&p) {
                continue;
            }
            let mut key_r = r.glwe_automorphism_key_alloc_from_infos(&k);
            fill_by_digit(r, &mut key_r, 1, &mut source);
            key_r.p = p;
            let mut key_t = t.glwe_automorphism_key_alloc_from_infos(&k);
            key_r.transfer_into(&mut key_t);
            let mut prep_r = r.glwe_automorphism_key_prepared_alloc_from_infos(&k);
            let mut prep_t = t.glwe_automorphism_key_prepared_alloc_from_infos(&k);
            r.glwe_automorphism_key_prepare(
                &mut prep_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_prepare_tmp_bytes(&k)).borrow(),
            );
            t.glwe_automorphism_key_prepare(
                &mut prep_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_prepare_tmp_bytes(&k)).borrow(),
            );
            keys_r.insert(p, prep_r);
            keys_t.insert(p, prep_t);
        }
        let a_r = ref_glwe(r, &g, &mut source);
        let mut a_t = t.glwe_alloc_from_infos(&g);
        a_r.transfer_into(&mut a_t);
        // An empty trace returns the input's coefficients without its noise estimate.
        let mut untagged = r.glwe_alloc_from_infos(&g);
        a_r.transfer_into(&mut untagged);
        untagged.noise = None;
        // Normalized forward traces onto every subring kind, then summed windows forward and reversed, and a
        // normalized reverse trace.
        let cases = [
            (first..end, false, true),
            (first + 1..end, false, true),
            (end..end, false, true),
            (1..3, false, false),
            (1..3, true, false),
            (2..end, true, true),
        ];
        for (levels, reverse, normalize) in cases {
            let label = format!("rank={rank} levels={levels:?} reverse={reverse} normalize={normalize}");
            let mut out_r = ref_glwe(r, &g, &mut source);
            let mut out_t = t.glwe_alloc_from_infos(&g);
            out_r.transfer_into(&mut out_t);
            r.glwe_trace(
                &mut out_r,
                levels.clone(),
                reverse,
                normalize,
                &a_r,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_trace_tmp_bytes(&g, &g, &k)).borrow(),
            );
            t.glwe_trace(
                &mut out_t,
                levels.clone(),
                reverse,
                normalize,
                &a_t,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_trace_tmp_bytes(&g, &g, &k)).borrow(),
            );
            let mut have = r.glwe_alloc_from_infos(&g);
            out_t.transfer_into(&mut have);
            assert_glwe_eq!(out_r, have, "trace {label}");
            if levels.is_empty() {
                assert_eq!(out_r, untagged, "empty out-of-place trace must copy the input");
            }
            a_r.transfer_into(&mut out_r);
            a_r.transfer_into(&mut out_t);
            r.glwe_trace_assign(
                &mut out_r,
                levels.clone(),
                reverse,
                normalize,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_trace_assign_tmp_bytes(&g, &k)).borrow(),
            );
            t.glwe_trace_assign(
                &mut out_t,
                levels.clone(),
                reverse,
                normalize,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_trace_assign_tmp_bytes(&g, &k)).borrow(),
            );
            out_t.transfer_into(&mut have);
            assert_glwe_eq!(out_r, have, "trace assign {label}");
            if levels.is_empty() {
                assert_eq!(out_r, untagged, "empty trace must preserve coefficients and layout");
            }
        }
        // Empty traces require no key lookup. Refused levels panic before changing the output, through either entry
        // point, and the key query refuses them alike.
        macro_rules! check_trace_edges {
            ($be:ty, $module:ident, $a:ident) => {{
                let empty_keys: HashMap<
                    i64,
                    crate::layouts::GLWEAutomorphismKeyPrepared<<$be as poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                > = HashMap::new();
                let initial = ref_glwe(r, &g, &mut source);
                let mut out = $module.glwe_alloc_from_infos(&g);
                initial.transfer_into(&mut out);
                let mut scratch = poisoned_scratch::<$be>($module.glwe_trace_tmp_bytes(&g, &g, &k));
                $module.glwe_trace(&mut out, end..end, false, true, &$a, &empty_keys, &mut scratch.borrow());
                let mut have = r.glwe_alloc_from_infos(&g);
                out.transfer_into(&mut have);
                assert_eq!(have, untagged, "empty trace without keys");
                for (levels, message) in &refused {
                    for assign in [false, true] {
                        initial.transfer_into(&mut out);
                        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if assign {
                                $module.glwe_trace_assign(
                                    &mut out,
                                    levels.clone(),
                                    false,
                                    true,
                                    &empty_keys,
                                    &mut scratch.borrow(),
                                );
                            } else {
                                $module.glwe_trace(
                                    &mut out,
                                    levels.clone(),
                                    false,
                                    true,
                                    &$a,
                                    &empty_keys,
                                    &mut scratch.borrow(),
                                );
                            }
                        }));
                        let Err(err) = rejected else {
                            panic!("trace accepted levels {levels:?}, assign={assign}");
                        };
                        assert!(
                            panic_message(&*err).contains(message),
                            "levels {levels:?}, assign={assign}: {}",
                            panic_message(&*err)
                        );
                        out.transfer_into(&mut have);
                        assert_eq!(
                            have, initial,
                            "refused levels {levels:?} changed the output, assign={assign}"
                        );
                    }
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        $module.glwe_trace_galois_elements(n, levels.clone(), false)
                    }));
                    let Err(err) = rejected else {
                        panic!("the trace key query accepted levels {levels:?}");
                    };
                    assert!(panic_message(&*err).contains(message), "key query levels {levels:?}");
                }
            }};
        }
        check_trace_edges!(BR, r, a_r);
        check_trace_edges!(BT, t, a_t);
    }
}

/// Covers packing, whose closing phase is a trace, over sparse packing trees that exercise both one-child branches and
/// the pair merge.
pub fn test_trace_packing_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETrace<BR> + GLWEPacking<BR> + GLWEAutomorphismKeyPreparedFactory<BR>,
    Module<BT>: GLWETrace<BT> + GLWEPacking<BT> + GLWEAutomorphismKeyPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let b = params.base2k;
    let mut source = Source::new([131; 32]);
    // The key lists name the module's Galois elements; keys for operands of a smaller
    // degree are those of a module of that degree.
    if r.n() == t.n() {
        assert_eq!(r.glwe_pack_galois_elements(), t.glwe_pack_galois_elements());
    }
    for &rank in &shapes.ranks {
        let g = GLWELayout {
            n: Degree(params.n as u32),
            base2k: Base2K(b as u32),
            k: TorusPrecision((2 * b + 1) as u32),
            rank: Rank(rank as u32),
        };
        // Covers the widest packing input below, `g_wide`.
        let k = GLWEAutomorphismKeyLayout {
            n: g.n,
            base2k: Base2K((b - 1) as u32),
            dnum: Dnum((5 * b).div_ceil(b - 1) as u32),
            k_digit: TorusPrecision((b - 1) as u32),
            k_aux: TorusPrecision(b as u32 + 1),
            rank: g.rank,
        };
        let mut keys_r = HashMap::new();
        let mut keys_t = HashMap::new();
        for p in r.glwe_pack_galois_elements() {
            let mut key_r = r.glwe_automorphism_key_alloc_from_infos(&k);
            fill_by_digit(r, &mut key_r, 1, &mut source);
            key_r.p = p;
            let mut key_t = t.glwe_automorphism_key_alloc_from_infos(&k);
            key_r.transfer_into(&mut key_t);
            let mut prep_r = r.glwe_automorphism_key_prepared_alloc_from_infos(&k);
            let mut prep_t = t.glwe_automorphism_key_prepared_alloc_from_infos(&k);
            r.glwe_automorphism_key_prepare(
                &mut prep_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_prepare_tmp_bytes(&k)).borrow(),
            );
            t.glwe_automorphism_key_prepare(
                &mut prep_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_prepare_tmp_bytes(&k)).borrow(),
            );
            keys_r.insert(p, prep_r);
            keys_t.insert(p, prep_t);
        }
        // Packing inputs need not share the destination layout: the merge tree and
        // the closing trace both run at the input layout, so a wider input is the
        // case a single-layout sweep cannot see.
        let g_wide = GLWELayout {
            k: TorusPrecision((5 * b) as u32),
            ..g
        };
        for (positions, log_gap_out, gi) in [
            (vec![0], 0, &g),
            (vec![0, params.n / 2, params.n - 1], 0, &g),
            (vec![0, 1, params.n / 2, params.n / 2 + 1], 0, &g),
            (vec![0, 2, 4], 1, &g),
            (vec![0, 2, 4], 1, &g_wide),
            (vec![0, params.n / 2, params.n - 1], 0, &g_wide),
        ] {
            let mut inputs_r: Vec<_> = positions.iter().map(|_| ref_glwe(r, gi, &mut source)).collect();
            let mut inputs_t: Vec<_> = inputs_r
                .iter()
                .map(|a| {
                    let mut x = t.glwe_alloc_from_infos(gi);
                    a.transfer_into(&mut x);
                    x
                })
                .collect();
            let map_r: HashMap<_, _> = positions.iter().copied().zip(inputs_r.iter_mut()).collect();
            let map_t: HashMap<_, _> = positions.iter().copied().zip(inputs_t.iter_mut()).collect();
            let mut out_r = ref_glwe(r, &g, &mut source);
            let mut out_t = t.glwe_alloc_from_infos(&g);
            out_r.transfer_into(&mut out_t);
            r.glwe_pack(
                &mut out_r,
                map_r,
                log_gap_out,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_pack_tmp_bytes(&g, gi, &k)).borrow(),
            );
            t.glwe_pack(
                &mut out_t,
                map_t,
                log_gap_out,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_pack_tmp_bytes(&g, gi, &k)).borrow(),
            );
            let mut have = r.glwe_alloc_from_infos(&g);
            out_t.transfer_into(&mut have);
            assert_glwe_eq!(out_r, have, "pack rank={rank} positions={positions:?} gap={log_gap_out}");
            if log_gap_out == 0 && gi.k == g.k {
                // The last packing phase is an empty trace: it must publish
                // the accumulator retained at index zero into the destination, untagged.
                let mut want = r.glwe_alloc_from_infos(&g);
                inputs_r[0].transfer_into(&mut want);
                want.noise = None;
                assert_eq!(out_r, want, "packing failed to publish its final accumulator");
            }
            // Inputs are explicitly consumed; compare the observable mutations too.
            let mut have_in = r.glwe_alloc_from_infos(gi);
            for (a, b) in inputs_r.iter().zip(inputs_t.iter()) {
                b.transfer_into(&mut have_in);
                assert_eq!(*a, have_in, "pack input mutation");
            }
        }
        macro_rules! check_invalid_pack {
            ($be:ty, $module:ident, $keys:ident) => {{
                for (positions, gap) in [
                    (vec![0, 1], 1),
                    (vec![1], 1),
                    (vec![0], (params.n.ilog2() as usize) + 1),
                ] {
                    let initial = ref_glwe(r, &g, &mut source);
                    let mut out = $module.glwe_alloc_from_infos(&g);
                    initial.transfer_into(&mut out);
                    let expected: Vec<_> = positions.iter().map(|_| ref_glwe(r, &g, &mut source)).collect();
                    let mut operands: Vec<_> = expected
                        .iter()
                        .map(|a| {
                            let mut x = $module.glwe_alloc_from_infos(&g);
                            a.transfer_into(&mut x);
                            x
                        })
                        .collect();
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let map = positions.iter().copied().zip(operands.iter_mut()).collect();
                        $module.glwe_pack(
                            &mut out,
                            map,
                            gap,
                            &$keys,
                            &mut poisoned_scratch::<$be>($module.glwe_pack_tmp_bytes(&g, &g, &k)).borrow(),
                        );
                    }));
                    assert!(rejected.is_err(), "packing accepted unsupported positions/gap");
                    let mut have = r.glwe_alloc_from_infos(&g);
                    out.transfer_into(&mut have);
                    assert_eq!(initial, have, "invalid packing changed output");
                    for (want, operand) in expected.iter().zip(&operands) {
                        operand.transfer_into(&mut have);
                        assert_eq!(*want, have, "invalid packing consumed an input");
                    }
                }
            }};
        }
        check_invalid_pack!(BR, r, keys_r);
        check_invalid_pack!(BT, t, keys_t);
        // Put each mismatched field at every actual HashMap iteration position.
        // This deterministically covers a wider input hidden after the layout used
        // for sizing, independently of the map's randomized order.
        macro_rules! check_mixed_pack_layouts {
            ($be:ty, $module:ident, $keys:ident) => {{
                for mixed in [
                    g_wide,
                    GLWELayout {
                        n: Degree(2 * g.n.0),
                        ..g
                    },
                    GLWELayout {
                        base2k: Base2K((b - 1) as u32),
                        ..g
                    },
                    GLWELayout {
                        rank: Rank(g.rank.0 + 1),
                        ..g
                    },
                ] {
                    let positions = [0, params.n / 2, params.n - 1];
                    for mixed_slot in 0..positions.len() {
                        let initial = ref_glwe(r, &g, &mut source);
                        let mut out = $module.glwe_alloc_from_infos(&g);
                        initial.transfer_into(&mut out);
                        let mut expected: Vec<_> = positions.iter().map(|_| ref_glwe(r, &g, &mut source)).collect();
                        let mut operands: Vec<_> = expected
                            .iter()
                            .map(|input| {
                                let mut operand = $module.glwe_alloc_from_infos(&g);
                                input.transfer_into(&mut operand);
                                operand
                            })
                            .collect();
                        let mut map: HashMap<_, _> = positions.iter().copied().zip(operands.iter_mut()).collect();
                        let (&position, input) = map.iter_mut().nth(mixed_slot).unwrap();
                        let operand_index = positions.iter().position(|&index| index == position).unwrap();
                        expected[operand_index] = ref_glwe(r, &mixed, &mut source);
                        **input = $module.glwe_alloc_from_infos(&mixed);
                        expected[operand_index].transfer_into(*input);
                        let mut scratch = poisoned_scratch::<$be>($module.glwe_pack_tmp_bytes(&g, &g, &k));
                        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            $module.glwe_pack(&mut out, map, 0, &$keys, &mut scratch.borrow());
                        }));
                        assert!(
                            rejected.is_err(),
                            "packing accepted mixed layouts: {mixed:?} slot={mixed_slot}"
                        );
                        let mut have = r.glwe_alloc_from_infos(&g);
                        out.transfer_into(&mut have);
                        assert_eq!(initial, have, "mixed packing layouts changed output");
                        for (want, operand) in expected.iter().zip(&operands) {
                            assert_eq!(
                                want.glwe_layout(),
                                operand.glwe_layout(),
                                "mixed packing layouts changed metadata"
                            );
                            let mut have = r.glwe_alloc_from_infos(want);
                            operand.transfer_into(&mut have);
                            assert_eq!(*want, have, "mixed packing layouts consumed an input");
                        }
                    }
                }
            }};
        }
        check_mixed_pack_layouts!(BR, r, keys_r);
        check_mixed_pack_layouts!(BT, t, keys_t);
    }
}

/// One-pass product roles of [`test_tensor_relinearize_decrypt_parity_roles`] that hold one product: a product, a
/// product reading its left operand from the destination and a squaring, the test side reading unnormalized twins (0
/// to 2) or canonical operands (6 to 8), and one prepared pair (5).
pub const ONE_PASS_PRODUCT_ROLES: [usize; 7] = [0, 1, 2, 5, 6, 7, 8];

/// One-pass product roles of [`test_tensor_relinearize_decrypt_parity_roles`] that sum three products: over
/// ciphertexts (3) and over pairs prepared once (4).
pub const ONE_PASS_SUM_ROLES: [usize; 2] = [3, 4];

/// Relinearization and tensor decryption use explicitly staged coefficient
/// inputs; secret preparation itself is compared before its prepared form is used.
pub fn test_tensor_relinearize_decrypt_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWEMulRelinearizeSum<BR>
        + GLWEPreparedTensoring<BR>
        + CnvPVecAlloc<BR>
        + Convolution<BR>
        + GLWENormalize<BR>
        + GLWETensorDecrypt<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWESecretTensorFactory<BR>
        + GLWESecretTensorPreparedFactory<BR>,
    Module<BT>: GLWETensoring<BT>
        + GLWEMulRelinearizeSum<BT>
        + GLWEPreparedTensoring<BT>
        + CnvPVecAlloc<BT>
        + Convolution<BT>
        + GLWENormalize<BT>
        + GLWETensorDecrypt<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + GLWESecretPreparedFactory<BT>
        + GLWESecretTensorFactory<BT>
        + GLWESecretTensorPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let roles: Vec<usize> = ONE_PASS_PRODUCT_ROLES.into_iter().chain(ONE_PASS_SUM_ROLES).collect();
    test_tensor_relinearize_decrypt_parity_roles(params, shapes, &roles, r, t);
}

/// [`test_tensor_relinearize_decrypt_parity`] on the given one-pass product roles only: a radix at which the oracle
/// holds one product may not hold a sum.
pub fn test_tensor_relinearize_decrypt_parity_roles<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    roles: &[usize],
    r: &Module<BR>,
    t: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWEMulRelinearizeSum<BR>
        + GLWEPreparedTensoring<BR>
        + CnvPVecAlloc<BR>
        + Convolution<BR>
        + GLWENormalize<BR>
        + GLWETensorDecrypt<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWESecretTensorFactory<BR>
        + GLWESecretTensorPreparedFactory<BR>,
    Module<BT>: GLWETensoring<BT>
        + GLWEMulRelinearizeSum<BT>
        + GLWEPreparedTensoring<BT>
        + CnvPVecAlloc<BT>
        + Convolution<BT>
        + GLWENormalize<BT>
        + GLWETensorDecrypt<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + GLWESecretPreparedFactory<BT>
        + GLWESecretTensorFactory<BT>
        + GLWESecretTensorPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let b = params.base2k;
    let mut source = Source::new([137; 32]);
    for &rank in &shapes.ranks {
        let secret_layout = GLWESecretLayout {
            n: Degree(params.n as u32),
            rank: Rank(rank as u32),
        };
        let mut secret_r = r.glwe_secret_alloc_from_infos(&secret_layout);
        for col in 0..rank {
            for x in secret_r.data.at_mut(col, 0) {
                *x = (source.next_i64().unsigned_abs() % 3) as i64 - 1;
            }
        }
        secret_r.dist = Distribution::TernaryProb(2.0 / 3.0);
        let mut secret_t = t.glwe_secret_alloc_from_infos(&secret_layout);
        secret_r.transfer_into(&mut secret_t);
        let mut secretp_r = r.glwe_secret_prepared_alloc_from_infos(&secret_r);
        r.glwe_secret_prepare(&mut secretp_r, &secret_r);
        let mut secretp_t = t.glwe_secret_prepared_alloc_from_infos(&secret_t);
        t.glwe_secret_prepare(&mut secretp_t, &secret_t);
        let mut tensor_r = r.glwe_secret_tensor_alloc_from_infos(&secret_layout);
        let mut tensor_t = t.glwe_secret_tensor_alloc_from_infos(&secret_layout);
        r.glwe_secret_tensor_prepare(
            &mut tensor_r,
            &secret_r,
            &mut poisoned_scratch::<BR>(r.glwe_secret_tensor_prepare_tmp_bytes(Rank(rank as u32))).borrow(),
        );
        crate::test_suite::assert_wipes_scratch::<BT>(t.glwe_secret_tensor_prepare_tmp_bytes(Rank(rank as u32)), |scratch| {
            t.glwe_secret_tensor_prepare(&mut tensor_t, &secret_t, scratch)
        });
        assert_eq!(
            poulpy_hal::test_suite::download_scalar_znx::<BR>(&tensor_r.data),
            poulpy_hal::test_suite::download_scalar_znx::<BT>(&tensor_t.data),
            "tensor secret rank={rank}"
        );
        assert_eq!(tensor_r.dist(), tensor_t.dist());
        assert_eq!(tensor_r.dist(), secret_r.dist());
        let mut tensorp_r = r.glwe_secret_tensor_prepared_alloc_from_infos(&secret_layout);
        r.glwe_secret_tensor_prepared_prepare(&mut tensorp_r, &tensor_r);
        let mut tensorp_t = t.glwe_secret_tensor_prepared_alloc_from_infos(&secret_layout);
        t.glwe_secret_tensor_prepared_prepare(&mut tensorp_t, &tensor_t);
        for precision in [2 * b, 2 * b + 1] {
            let g = GLWELayout {
                n: Degree(params.n as u32),
                base2k: Base2K(b as u32),
                k: TorusPrecision(precision as u32),
                rank: Rank(rank as u32),
            };
            let mut a_r = r.glwe_tensor_alloc_from_infos(&g);
            r.fill_glwe_from_source(&mut a_r, &mut source);
            let mut a_t = t.glwe_tensor_alloc_from_infos(&g);
            a_r.transfer_into(&mut a_t);
            let mut pt_r = r.glwe_plaintext_alloc_from_infos(&g);
            r.fill_glwe_from_source(&mut pt_r, &mut source);
            let mut pt_t = t.glwe_plaintext_alloc_from_infos(&g);
            pt_r.transfer_into(&mut pt_t);
            r.glwe_tensor_decrypt(
                &a_r,
                &mut pt_r,
                &secretp_r,
                &tensorp_r,
                &mut poisoned_scratch::<BR>(r.glwe_tensor_decrypt_tmp_bytes(&g)).borrow(),
            );
            t.glwe_tensor_decrypt(
                &a_t,
                &mut pt_t,
                &secretp_t,
                &tensorp_t,
                &mut poisoned_scratch::<BT>(t.glwe_tensor_decrypt_tmp_bytes(&g)).borrow(),
            );
            let mut have = r.glwe_plaintext_alloc_from_infos(&g);
            pt_t.transfer_into(&mut have);
            assert_eq!((pt_r.base2k, pt_r.k), (have.base2k, have.k));
            assert_eq!(
                poulpy_hal::test_suite::download_vec_znx::<BR>(&pt_r.data),
                poulpy_hal::test_suite::download_vec_znx::<BR>(&have.data),
                "tensor decrypt rank={rank} k={precision}"
            );
            for k_digit in shapes.k_digits(precision, b, rank) {
                let key = GLWETensorKeyLayout {
                    n: g.n,
                    base2k: g.base2k,
                    dnum: Dnum(precision.div_ceil(k_digit) as u32),
                    k_digit: TorusPrecision(k_digit as u32),
                    k_aux: TorusPrecision((k_digit + 1) as u32),
                    rank: g.rank,
                };
                let mut key_r = r.glwe_tensor_key_alloc_from_infos(&key);
                fill_by_digit(r, &mut key_r, 1, &mut source);
                let mut key_t = t.glwe_tensor_key_alloc_from_infos(&key);
                key_r.transfer_into(&mut key_t);
                let mut kp_r = r.alloc_tensor_key_prepared_from_infos(&key);
                let mut kp_t = t.alloc_tensor_key_prepared_from_infos(&key);
                r.prepare_tensor_key(
                    &mut kp_r,
                    &key_r,
                    &mut poisoned_scratch::<BR>(r.prepare_tensor_key_tmp_bytes(&key)).borrow(),
                );
                t.prepare_tensor_key(
                    &mut kp_t,
                    &key_t,
                    &mut poisoned_scratch::<BT>(t.prepare_tensor_key_tmp_bytes(&key)).borrow(),
                );
                let mut out_r = ref_glwe(r, &g, &mut source);
                let mut out_t = t.glwe_alloc_from_infos(&g);
                out_r.transfer_into(&mut out_t);
                r.glwe_tensor_relinearize(
                    &mut out_r,
                    &a_r,
                    &kp_r,
                    &mut poisoned_scratch::<BR>(r.glwe_tensor_relinearize_tmp_bytes(&g, &g, &key)).borrow(),
                );
                t.glwe_tensor_relinearize(
                    &mut out_t,
                    &a_t,
                    &kp_t,
                    &mut poisoned_scratch::<BT>(t.glwe_tensor_relinearize_tmp_bytes(&g, &g, &key)).borrow(),
                );
                let mut have = r.glwe_alloc_from_infos(&g);
                out_t.transfer_into(&mut have);
                assert_glwe_eq!(out_r, have, "relinearize rank={rank} k={precision} k_digit={k_digit}");

                let wide = GLWELayout {
                    k: TorusPrecision(key.gglwe_layout().max_input_k().0 + 1),
                    ..g
                };
                let wide_tensor = t.glwe_tensor_alloc_from_infos(&wide);
                assert_refuses_uncovered_input("glwe_tensor_relinearize took a tensor one bit past its key", || {
                    t.glwe_tensor_relinearize(
                        &mut out_t,
                        &wide_tensor,
                        &kp_t,
                        &mut poisoned_scratch::<BT>(t.glwe_tensor_relinearize_tmp_bytes(&g, &wide, &key)).borrow(),
                    )
                });
                if rank == 1 {
                    let wide_glwe = t.glwe_alloc_from_infos(&wide);
                    let size = wide.k.as_usize().div_ceil(b);
                    assert_refuses_uncovered_input("glwe_mul_relinearize took operands one bit past its key", || {
                        t.glwe_mul_relinearize(
                            b,
                            &mut out_t,
                            g.k,
                            &wide_glwe,
                            &wide_glwe,
                            &kp_t,
                            &mut poisoned_scratch::<BT>(t.glwe_mul_relinearize_tmp_bytes(&g, size, size, wide.k, &key)).borrow(),
                        )
                    });
                }

                // The one-pass product must agree byte for byte across backends, in each operand role.
                if rank == 1 {
                    let x_r = ref_glwe(r, &g, &mut source);
                    let y_r = ref_glwe(r, &g, &mut source);
                    let mut x_t = t.glwe_alloc_from_infos(&g);
                    x_r.transfer_into(&mut x_t);
                    let mut y_t = t.glwe_alloc_from_infos(&g);
                    y_r.transfer_into(&mut y_t);
                    let mut x_twin_t = t.glwe_alloc_from_infos(&g);
                    unnormalized_twin::<BR, BT>(&x_r, &mut x_twin_t);
                    let mut y_twin_t = t.glwe_alloc_from_infos(&g);
                    unnormalized_twin::<BR, BT>(&y_r, &mut y_twin_t);
                    let size = precision.div_ceil(b);
                    // Role 3's pairs prepared once, under both hints on each side, each on exactly its poisoned budget.
                    macro_rules! prepared_pairs {
                        ($be:ty, $m:expr, $pairs:expr) => {{
                            $pairs
                                .iter()
                                .enumerate()
                                .map(|(i, (l, r))| {
                                    let (l_hint, r_hint) = crate::test_suite::prepare_hints(i);
                                    let mut l_prep = $m.cnv_pvec_left_alloc(params.n, 2, size, l_hint);
                                    glwe_prepare_left(
                                        $m,
                                        &mut l_prep,
                                        *l,
                                        precision,
                                        &mut poisoned_scratch::<$be>(glwe_prepare_left_tmp_bytes($m, size, &g)).borrow(),
                                    );
                                    let mut r_prep = $m.cnv_pvec_right_alloc(params.n, 2, size, r_hint);
                                    glwe_prepare_right(
                                        $m,
                                        &mut r_prep,
                                        *r,
                                        precision,
                                        &mut poisoned_scratch::<$be>(glwe_prepare_right_tmp_bytes($m, size, &g)).borrow(),
                                    );
                                    (l_prep, r_prep)
                                })
                                .collect::<Vec<_>>()
                        }};
                    }
                    let prep_r = prepared_pairs!(BR, r, [(&x_r, &y_r), (&y_r, &x_r), (&x_r, &x_r)]);
                    let prep_t = prepared_pairs!(BT, t, [(&x_t, &y_t), (&y_twin_t, &x_twin_t), (&x_t, &x_t)]);
                    let prep_r: (Vec<_>, Vec<_>) = prep_r.iter().map(|(l, r)| (l, r)).unzip();
                    let prep_t: (Vec<_>, Vec<_>) = prep_t.iter().map(|(l, r)| (l, r)).unzip();
                    for cnv_offset in [b, precision] {
                        let mut outs = std::collections::HashMap::new();
                        for &role in roles {
                            let mut out_r = r.glwe_alloc_from_infos(&g);
                            x_r.transfer_into(&mut out_r);
                            let mut out_t = t.glwe_alloc_from_infos(&g);
                            out_r.transfer_into(&mut out_t);
                            // `$x2` and `$y2` hold the values of `$x` and `$y`: the test side passes unnormalized twins.
                            macro_rules! one_pass {
                                ($be:ty, $m:expr, $out:expr, $x:expr, $y:expr, $x2:expr, $y2:expr, $prep:expr, $key:expr) => {{
                                    let (op, x2, y2) = if role >= 6 {
                                        (role - 6, $x, $y)
                                    } else {
                                        (role, $x2, $y2)
                                    };
                                    let bytes = match op {
                                        3 => $m.glwe_mul_relinearize_sum_tmp_bytes(&g, size, size, g.k, &key),
                                        4 => $m.glwe_mul_prepared_relinearize_sum_tmp_bytes(&g, size, size, g.k, &key),
                                        _ => $m.glwe_mul_relinearize_tmp_bytes(&g, size, size, g.k, &key),
                                    };
                                    let mut scratch = poisoned_scratch::<$be>(bytes);
                                    // Role 0 is a product, role 1 reads its left operand from the destination,
                                    // role 2 is a squaring, roles 6 to 8 the same on canonical operands, role 3 a
                                    // sum of three products whose second pair the test side reads from the twins,
                                    // role 4 the same sum over pairs prepared once, the test side's second pair
                                    // prepared from the twins, and role 5 the first prepared pair alone, on a single
                                    // product's scratch.
                                    match op {
                                        1 => {
                                            $m.glwe_mul_relinearize_assign(cnv_offset, $out, g.k, y2, $key, &mut scratch.borrow())
                                        }
                                        2 => $m.glwe_square_relinearize(cnv_offset, $out, g.k, x2, $key, &mut scratch.borrow()),
                                        3 => $m.glwe_mul_relinearize_sum(
                                            cnv_offset,
                                            $out,
                                            g.k,
                                            &[$x, $y2, $x],
                                            &[$y, $x2, $x],
                                            $key,
                                            &mut scratch.borrow(),
                                        ),
                                        4 | 5 => $m.glwe_mul_prepared_relinearize_sum(
                                            cnv_offset,
                                            $out,
                                            g.k,
                                            &$prep.0[..if role == 5 { 1 } else { 3 }],
                                            &$prep.1[..if role == 5 { 1 } else { 3 }],
                                            size,
                                            g.k,
                                            size,
                                            g.k,
                                            $key,
                                            &mut scratch.borrow(),
                                        ),
                                        _ => $m.glwe_mul_relinearize(cnv_offset, $out, g.k, x2, y2, $key, &mut scratch.borrow()),
                                    }
                                }};
                            }
                            one_pass!(BR, r, &mut out_r, &x_r, &y_r, &x_r, &y_r, &prep_r, &kp_r);
                            one_pass!(BT, t, &mut out_t, &x_t, &y_t, &x_twin_t, &y_twin_t, &prep_t, &kp_t);
                            let mut have = r.glwe_alloc_from_infos(&g);
                            out_t.transfer_into(&mut have);
                            assert_glwe_eq!(
                                out_r,
                                have,
                                "one-pass product role={role} k={precision} k_digit={k_digit} offset={cnv_offset}"
                            );
                            outs.insert(role, out_r);
                        }
                        // Prepared pairs give the sum, and the first alone the product, bit for bit.
                        if let (Some(sum), Some(prepared)) = (outs.get(&3), outs.get(&4)) {
                            assert_glwe_eq!(
                                *sum,
                                *prepared,
                                "one-pass sum versus prepared pairs k={precision} k_digit={k_digit} offset={cnv_offset}"
                            );
                        }
                        if let (Some(product), Some(prepared)) = (outs.get(&0), outs.get(&5)) {
                            assert_glwe_eq!(
                                *product,
                                *prepared,
                                "one-pass product versus one prepared pair k={precision} k_digit={k_digit} offset={cnv_offset}"
                            );
                        }
                    }
                }
            }
        }
    }
}
