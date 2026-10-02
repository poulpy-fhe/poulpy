//! Trace, packing, relinearization and tensor-secret parity.
use super::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe};
use crate::{
    Distribution, GLWEMaskFill, GLWEPacking, GLWETensorDecrypt, GLWETensoring, GLWETrace, GetDistribution,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWESecretTensorFactory,
        GLWETensorKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretTensorPreparedFactory,
            GLWETensorKeyPreparedFactory,
        },
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};
use std::collections::HashMap;

/// Covers all trace and packing variants, complete/partial traces, and sparse
/// packing trees that exercise both one-child branches and the pair merge.
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
    assert_eq!(r.glwe_trace_galois_elements(), t.glwe_trace_galois_elements());
    assert_eq!(r.glwe_pack_galois_elements(), t.glwe_pack_galois_elements());
    for &rank in &shapes.ranks {
        let g = GLWELayout {
            n: Degree(r.n() as u32),
            base2k: Base2K(b as u32),
            k: TorusPrecision((2 * b + 1) as u32),
            rank: Rank(rank as u32),
        };
        let k = GLWEAutomorphismKeyLayout {
            n: g.n,
            base2k: Base2K((b - 1) as u32),
            dnum: Dnum((2 * b + 1).div_ceil(b - 1) as u32),
            dsize: Dsize(1),
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
        let a_r = ref_glwe(r, &g, &mut source);
        let mut a_t = t.glwe_alloc_from_infos(&g);
        a_r.transfer_into(&mut a_t);
        for skip in [0, 1, r.log_n()] {
            let mut out_r = ref_glwe(r, &g, &mut source);
            let mut out_t = t.glwe_alloc_from_infos(&g);
            out_r.transfer_into(&mut out_t);
            r.glwe_trace(
                &mut out_r,
                skip,
                &a_r,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_trace_tmp_bytes(&g, &g, &k)).borrow(),
            );
            t.glwe_trace(
                &mut out_t,
                skip,
                &a_t,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_trace_tmp_bytes(&g, &g, &k)).borrow(),
            );
            let mut have = r.glwe_alloc_from_infos(&g);
            out_t.transfer_into(&mut have);
            assert_glwe_eq!(out_r, have, "trace rank={rank} skip={skip}");
            if skip == r.log_n() {
                assert_eq!(out_r, a_r, "empty out-of-place trace must copy the input");
            }
            a_r.transfer_into(&mut out_r);
            a_r.transfer_into(&mut out_t);
            r.glwe_trace_assign(
                &mut out_r,
                skip,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_trace_assign_tmp_bytes(&g, &k)).borrow(),
            );
            t.glwe_trace_assign(
                &mut out_t,
                skip,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_trace_assign_tmp_bytes(&g, &k)).borrow(),
            );
            out_t.transfer_into(&mut have);
            assert_glwe_eq!(out_r, have, "trace assign rank={rank} skip={skip}");
            if skip == r.log_n() {
                assert_eq!(out_r, a_r, "empty trace must preserve all input coefficients and metadata");
            }
        }
        // Empty traces require no key lookup. Invalid skips reject before
        // changing output, consistently with the assign entry point.
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
                $module.glwe_trace(&mut out, $module.log_n(), &$a, &empty_keys, &mut scratch.borrow());
                let mut have = r.glwe_alloc_from_infos(&g);
                out.transfer_into(&mut have);
                assert_eq!(have, a_r, "empty trace without keys");
                for assign in [false, true] {
                    initial.transfer_into(&mut out);
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if assign {
                            $module.glwe_trace_assign(&mut out, $module.log_n() + 1, &empty_keys, &mut scratch.borrow());
                        } else {
                            $module.glwe_trace(&mut out, $module.log_n() + 1, &$a, &empty_keys, &mut scratch.borrow());
                        }
                    }));
                    assert!(rejected.is_err(), "trace accepted skip greater than log_n");
                    out.transfer_into(&mut have);
                    assert_eq!(have, initial, "invalid skip changed output");
                }
            }};
        }
        check_trace_edges!(BR, r, a_r);
        check_trace_edges!(BT, t, a_t);
        // Packing inputs need not share the destination layout: the merge tree and
        // the closing trace both run at the input layout, so a wider input is the
        // case a single-layout sweep cannot see.
        let g_wide = GLWELayout {
            k: TorusPrecision((5 * b) as u32),
            ..g
        };
        for (positions, log_gap_out, gi) in [
            (vec![0], 0, &g),
            (vec![0, r.n() / 2, r.n() - 1], 0, &g),
            (vec![0, 1, r.n() / 2, r.n() / 2 + 1], 0, &g),
            (vec![0, 2, 4], 1, &g),
            (vec![0, 2, 4], 1, &g_wide),
            (vec![0, r.n() / 2, r.n() - 1], 0, &g_wide),
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
                // the accumulator retained at index zero into the destination.
                assert_eq!(out_r, inputs_r[0], "packing failed to publish its final accumulator");
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
                for (positions, gap) in [(vec![0, 1], 1), (vec![1], 1), (vec![0], $module.log_n() + 1)] {
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
                    let positions = [0, $module.n() / 2, $module.n() - 1];
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

/// Relinearization and tensor decryption use explicitly staged coefficient
/// inputs; secret preparation itself is compared before its prepared form is used.
pub fn test_tensor_relinearize_decrypt_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWETensorDecrypt<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWESecretTensorFactory<BR>
        + GLWESecretTensorPreparedFactory<BR>,
    Module<BT>: GLWETensoring<BT>
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
        let mut secret_r = r.glwe_secret_alloc(Rank(rank as u32));
        for col in 0..rank {
            for x in secret_r.data.at_mut(col, 0) {
                *x = (source.next_i64().unsigned_abs() % 3) as i64 - 1;
            }
        }
        secret_r.dist = Distribution::TernaryProb(2.0 / 3.0);
        let mut secret_t = t.glwe_secret_alloc(Rank(rank as u32));
        secret_r.transfer_into(&mut secret_t);
        let mut secretp_r = r.glwe_secret_prepared_alloc_from_infos(&secret_r);
        r.glwe_secret_prepare(&mut secretp_r, &secret_r);
        let mut secretp_t = t.glwe_secret_prepared_alloc_from_infos(&secret_t);
        t.glwe_secret_prepare(&mut secretp_t, &secret_t);
        let mut tensor_r = r.glwe_secret_tensor_alloc(Rank(rank as u32));
        let mut tensor_t = t.glwe_secret_tensor_alloc(Rank(rank as u32));
        r.glwe_secret_tensor_prepare(
            &mut tensor_r,
            &secret_r,
            &mut poisoned_scratch::<BR>(r.glwe_secret_tensor_prepare_tmp_bytes(Rank(rank as u32))).borrow(),
        );
        t.glwe_secret_tensor_prepare(
            &mut tensor_t,
            &secret_t,
            &mut poisoned_scratch::<BT>(t.glwe_secret_tensor_prepare_tmp_bytes(Rank(rank as u32))).borrow(),
        );
        assert_eq!(
            BR::to_host_bytes(&tensor_r.data.data),
            BT::to_host_bytes(&tensor_t.data.data),
            "tensor secret rank={rank}"
        );
        assert_eq!(tensor_r.dist(), tensor_t.dist());
        assert_eq!(tensor_r.dist(), secret_r.dist());
        let mut tensorp_r = r.glwe_secret_tensor_prepared_alloc(Rank(rank as u32));
        r.glwe_secret_tensor_prepared_prepare(&mut tensorp_r, &tensor_r);
        let mut tensorp_t = t.glwe_secret_tensor_prepared_alloc(Rank(rank as u32));
        t.glwe_secret_tensor_prepared_prepare(&mut tensorp_t, &tensor_t);
        for precision in [2 * b, 2 * b + 1] {
            let g = GLWELayout {
                n: Degree(r.n() as u32),
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
                BR::to_host_bytes(pt_r.data.data()),
                BR::to_host_bytes(have.data.data()),
                "tensor decrypt rank={rank} k={precision}"
            );
            for dsize in shapes.dsizes(precision, b) {
                let key = GLWETensorKeyLayout {
                    n: g.n,
                    base2k: g.base2k,
                    dnum: Dnum(precision.div_ceil(b * dsize) as u32),
                    dsize: Dsize(dsize as u32),
                    k_aux: TorusPrecision((b * dsize + 1) as u32),
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
                assert_glwe_eq!(out_r, have, "relinearize rank={rank} k={precision} dsize={dsize}");
            }
        }
    }
}
