//! Key lifecycle and packed-to-prepared integer parity.
use super::{ParityBackend, assert_panics_with, fixture_glwe, gglwe_prepared_bytes, snapshot_glwe};
use crate::{
    api::{
        BDDKeyEncryptSk, BDDKeyPreparedFactory, BlindRotationKeyEncryptSk, BlindRotationKeyPreparedFactory,
        CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPreparedFactory, ExecuteBDDCircuit1WTo1W,
        ExecuteBDDCircuit2WTo1W, FheUintPrepare, FheUintPreparedEncryptSk,
    },
    bdd_arithmetic::{
        BDDKey, BDDKeyInfos, BDDKeyLayout, BDDKeyPrepared, FheUint, FheUintPrepared, FheUintPreparedFactory, GetBitCircuitInfo,
        Node,
    },
    blind_rotation::{BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};
use poulpy_core::test_suite::with_scratch;
use poulpy_core::{
    GGLWEToGGSWKeyEncryptSk, GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEExternalProduct, GLWESwitchingKeyEncryptSk,
    GLWEToLWEKeyEncryptSk, TransferInto,
    api::{
        GGLWEToGGSWKeyPreparedFactory, GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory,
        GLWESecretSampling, GLWESwitchingKeyPreparedFactory, GLWEToLWEKeyPreparedFactory, LWESecretSampling,
    },
    layouts::{
        GGLWEToGGSWKeyLayout, GGSWInfos, GGSWLayout, GGSWPreparedToBackendRef, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout,
        GLWESecretAlloc, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout, LWEInfos, LWESecretAlloc,
    },
};
use poulpy_hal::{
    layouts::{HostBytesBackend, Module, WriterTo},
    source::Source,
};

struct SelectBits {
    nodes: [[Node; 2]; 8],
    two_words: bool,
}
impl SelectBits {
    fn new(two_words: bool) -> Self {
        Self {
            nodes: std::array::from_fn(|i| [Node::Cmux(i + if two_words && i % 2 != 0 { 8 } else { 0 }, 1, 0), Node::None]),
            two_words,
        }
    }
}
impl GetBitCircuitInfo for SelectBits {
    fn input_size(&self) -> usize {
        if self.two_words { 16 } else { 8 }
    }
    fn output_size(&self) -> usize {
        8
    }
    fn get_circuit(&self, bit: usize) -> (&[Node], usize) {
        (&self.nodes[bit], 2)
    }
}

fn key_layout(n: usize, bridge: bool) -> BDDKeyLayout {
    let rank = if bridge { 2usize } else { 1 };
    let aux = n.ilog2() as usize;
    BDDKeyLayout {
        cbt_layout: CircuitBootstrappingKeyLayout {
            brk_layout: BlindRotationKeyLayout {
                n_glwe: n.into(),
                n_lwe: 6usize.into(),
                base2k: 13usize.into(),
                dnum: 3usize.into(),
                k_aux: (13 + aux).into(),
                rank: rank.into(),
            },
            atk_layout: GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: 11usize.into(),
                dnum: 4usize.into(),
                k_digit: 11usize.into(),
                k_aux: (11 + aux).into(),
                rank: rank.into(),
            },
            // Covers the output GGSW, which the row expansion switches.
            tsk_layout: GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: 12usize.into(),
                dnum: 5usize.into(),
                k_digit: 12usize.into(),
                k_aux: (12 + aux).into(),
                rank: rank.into(),
            },
        },
        ks_glwe_layout: bridge.then_some(GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: 12usize.into(),
            dnum: 4usize.into(),
            k_digit: 12usize.into(),
            k_offset: poulpy_core::layouts::TorusPrecision(0),
            k_aux: (12 + aux).into(),
            rank_in: rank.into(),
            rank_out: 1usize.into(),
        }),
        ks_lwe_layout: GLWEToLWEKeyLayout {
            n: n.into(),
            base2k: 11usize.into(),
            dnum: 4usize.into(),
            k_aux: (11 + aux).into(),
            rank_in: 1usize.into(),
        },
    }
}

/// The byte query counts the allocated components, and both forms refuse a
/// switching key of another degree than the module's.
fn check_prepared_allocation<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &BDDKeyLayout,
    key: &BDDKeyPrepared<BE::OwnedBuf, CGGI, BE>,
) where
    Module<BE>: BDDKeyPreparedFactory<CGGI, BE>
        + CircuitBootstrappingKeyPreparedFactory<CGGI, BE>
        + BlindRotationKeyPreparedFactory<CGGI, BE>
        + GGSWPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWEToLWEKeyPreparedFactory<BE>,
{
    let bytes = super::circuit_bootstrapping::prepared_key_bytes(&key.cbt)
        + gglwe_prepared_bytes::<BE, _>(&key.ks_lwe)
        + key.ks_glwe.as_ref().map_or(0, gglwe_prepared_bytes::<BE, _>);
    assert_eq!(module.bdd_key_prepared_bytes_of_from_infos(layout), bytes);
    let half = (module.n() / 2).into();
    let mut foreign = vec![BDDKeyLayout {
        ks_lwe_layout: GLWEToLWEKeyLayout {
            n: half,
            ..layout.ks_lwe_layout
        },
        ..*layout
    }];
    if let Some(ks_glwe) = layout.ks_glwe_layout {
        foreign.push(BDDKeyLayout {
            ks_glwe_layout: Some(GLWESwitchingKeyLayout { n: half, ..ks_glwe }),
            ..*layout
        });
    }
    for foreign in &foreign {
        assert_panics_with(
            "bdd_key_prepared_alloc_from_infos: key degree differs from the module degree",
            || module.bdd_key_prepared_alloc_from_infos(foreign),
        );
        assert_panics_with(
            "bdd_key_prepared_bytes_of_from_infos: key degree differs from the module degree",
            || module.bdd_key_prepared_bytes_of_from_infos(foreign),
        );
    }
}

fn raw_key_bytes<BE: ParityBackend>(
    module: &Module<BE>,
    key: &BDDKey<BE::OwnedBuf, CGGI, i64>,
    layout: &BDDKeyLayout,
) -> Vec<u8> {
    let host = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut copy = BDDKey::alloc_from_infos(&host, layout);
    key.transfer_into(&mut copy);
    let mut bytes = Vec::new();
    copy.write_to(&mut bytes).unwrap();
    bytes
}

fn observe<BE: ParityBackend>(
    module: &Module<BE>,
    value: &FheUintPrepared<BE::OwnedBuf, u8, BE>,
) -> (GGSWLayout, Vec<super::GlweSnapshot>)
where
    Module<BE>: GLWEExternalProduct<BE>,
{
    let layout = GLWELayout {
        n: value.n(),
        base2k: 12usize.into(),
        k: 29usize.into(),
        rank: value.rank(),
    };
    let input = fixture_glwe(module, &layout, 53);
    (
        value.ggsw_layout(),
        value
            .bits
            .iter()
            .map(|bit| {
                let mut output = fixture_glwe(module, &layout, 59);
                with_scratch::<BE, _>(module.glwe_external_product_tmp_bytes(&output, &input, bit), |s| {
                    module.glwe_external_product(&mut output, &input, &bit.to_backend_ref(), s)
                });
                snapshot_glwe::<BE, _>(&output)
            })
            .collect(),
    )
}

/// Transfers a single raw BDD key and packed input, prepares independently, and
/// checks every prepared bit through a coefficient-domain external product.
/// Covers optional rank switching, full/custom ranges, empty ranges, reuse, and
/// supported worker counts. Prepared representations are never compared as bytes.
pub fn test_lifecycle_parity<BR: ParityBackend, BT: ParityBackend>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: BDDKeyEncryptSk<CGGI, BR>
        + BDDKeyPreparedFactory<CGGI, BR>
        + CircuitBootstrappingKeyPreparedFactory<CGGI, BR>
        + BlindRotationKeyPreparedFactory<CGGI, BR>
        + GGLWEToGGSWKeyPreparedFactory<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWESwitchingKeyPreparedFactory<BR>
        + GLWEToLWEKeyPreparedFactory<BR>
        + FheUintPrepare<CGGI, BR>
        + GGSWPreparedFactory<BR>
        + GLWEExternalProduct<BR>
        + GLWESecretSampling<BR>
        + LWESecretSampling<BR>
        + ExecuteBDDCircuit1WTo1W<BR>
        + ExecuteBDDCircuit2WTo1W<BR>,
    Module<BT>: BDDKeyPreparedFactory<CGGI, BT>
        + CircuitBootstrappingKeyPreparedFactory<CGGI, BT>
        + BlindRotationKeyPreparedFactory<CGGI, BT>
        + GGLWEToGGSWKeyPreparedFactory<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWESwitchingKeyPreparedFactory<BT>
        + GLWEToLWEKeyPreparedFactory<BT>
        + FheUintPrepare<CGGI, BT>
        + GGSWPreparedFactory<BT>
        + GLWEExternalProduct<BT>
        + ExecuteBDDCircuit1WTo1W<BT>
        + ExecuteBDDCircuit2WTo1W<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    for bridge in [false, true] {
        let layout = key_layout(module_ref.n(), bridge);
        let rank = layout.cbt_layout.brk_layout.rank;
        let output = GGSWLayout {
            n: module_ref.n().into(),
            base2k: 15usize.into(),
            dnum: 2usize.into(),
            k_digit: 15usize.into(),
            k_aux: (15 + module_ref.n().ilog2() as usize).into(),
            rank,
        };
        let mut sk = module_ref.glwe_secret_alloc(rank);
        module_ref.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([61; 32]));
        let mut sk_lwe = module_ref.lwe_secret_alloc(6usize.into());
        module_ref.lwe_secret_fill_binary_block(&mut sk_lwe, 3, &mut Source::new([67; 32]));
        let mut raw_ref = BDDKey::alloc_from_infos(module_ref, &layout);

        with_scratch::<BR, _>(module_ref.bdd_key_encrypt_sk_tmp_bytes(&layout), |s| {
            module_ref.bdd_key_encrypt_sk(
                &mut raw_ref,
                &sk_lwe,
                &sk,
                &mut Source::new([71; 32]),
                &mut Source::new([73; 32]),
                s,
            )
        });
        let mut raw_test = BDDKey::alloc_from_infos(module_test, &layout);
        raw_ref.transfer_into(&mut raw_test);
        assert!(
            raw_key_bytes(module_ref, &raw_ref, &layout) == raw_key_bytes(module_test, &raw_test, &layout),
            "raw key transfer mismatch"
        );
        let mut key_ref = module_ref.bdd_key_prepared_alloc_from_infos(&layout);
        let mut key_test = module_test.bdd_key_prepared_alloc_from_infos(&layout);
        check_prepared_allocation(module_ref, &layout, &key_ref);
        check_prepared_allocation(module_test, &layout, &key_test);
        with_scratch::<BR, _>(module_ref.bdd_key_prepare_tmp_bytes(&layout), |s| {
            module_ref.bdd_key_prepare(&mut key_ref, &raw_ref, s)
        });
        with_scratch::<BT, _>(module_test.bdd_key_prepare_tmp_bytes(&layout), |s| {
            module_test.bdd_key_prepare(&mut key_test, &raw_test, s)
        });
        assert_eq!(key_ref.cbt_infos().brk_layout, key_test.cbt_infos().brk_layout);
        assert_eq!(key_ref.ks_lwe_infos(), key_test.ks_lwe_infos());
        assert_eq!(key_ref.ks_glwe_infos(), key_test.ks_glwe_infos());
        let bits_layout = GLWELayout {
            n: module_ref.n().into(),
            base2k: 14usize.into(),
            k: 31usize.into(),
            rank,
        };
        let input_ref = FheUint::<_, u8, i64> {
            bits: fixture_glwe(module_ref, &bits_layout, 79),
            _phantom: std::marker::PhantomData,
        };
        let mut input_test = FheUint::<_, u8, i64>::alloc_from_infos(module_test, &bits_layout);
        input_ref.transfer_into(&mut input_test);
        let mut res_ref = module_ref.alloc_fhe_uint_prepared_from_infos(&output);
        let mut res_test = module_test.alloc_fhe_uint_prepared_from_infos(&output);
        let br_bytes = module_ref.fhe_uint_prepare_tmp_bytes(3, 1, &output, &bits_layout, &layout);
        let bt_bytes = module_test.fhe_uint_prepare_tmp_bytes(3, 1, &output, &bits_layout, &layout);
        with_scratch::<BR, _>(br_bytes, |s| {
            module_ref.fhe_uint_prepare(&mut res_ref, &input_ref, &key_ref, s)
        });
        with_scratch::<BT, _>(bt_bytes, |s| {
            module_test.fhe_uint_prepare(&mut res_test, &input_test, &key_test, s)
        });
        assert_eq!(observe(module_ref, &res_ref), observe(module_test, &res_test));
        for (start, count) in [(1usize, 5usize), (2, 0), (0, 8)] {
            with_scratch::<BR, _>(br_bytes, |s| {
                module_ref.fhe_uint_prepare_custom(&mut res_ref, &input_ref, start, count, &key_ref, s)
            });
            with_scratch::<BT, _>(bt_bytes, |s| {
                module_test.fhe_uint_prepare_custom(&mut res_test, &input_test, start, count, &key_test, s)
            });
            let expected = observe(module_ref, &res_ref);
            assert_eq!(expected, observe(module_test, &res_test));
            for threads in [1usize, 2, 4] {
                let workers = poulpy_hal::execution::worker_count::<BT::TaskExecutor>(threads, count);
                with_scratch::<BT, _>(bt_bytes * workers, |s| {
                    module_test.fhe_uint_prepare_custom_multi_thread(
                        threads,
                        &mut res_test,
                        &input_test,
                        start,
                        count,
                        &key_test,
                        s,
                    )
                });
                assert_eq!(expected, observe(module_test, &res_test));
            }
        }
        let mut right_ref = module_ref.alloc_fhe_uint_prepared_from_infos(&output);
        let mut right_test = module_test.alloc_fhe_uint_prepared_from_infos(&output);
        with_scratch::<BR, _>(br_bytes, |s| {
            module_ref.fhe_uint_prepare_custom(&mut right_ref, &input_ref, 0, 0, &key_ref, s)
        });
        with_scratch::<BT, _>(bt_bytes, |s| {
            module_test.fhe_uint_prepare_custom(&mut right_test, &input_test, 0, 0, &key_test, s)
        });
        for two_words in [false, true] {
            let circuit = SelectBits::new(two_words);
            let mut expected = FheUint::<_, u8, i64>::alloc_from_infos(module_ref, &bits_layout);
            let mut actual = FheUint::<_, u8, i64>::alloc_from_infos(module_test, &bits_layout);
            let rb = if two_words {
                module_ref.execute_bdd_circuit_2w_to_1w_tmp_bytes::<_, u8, _, _, _>(&bits_layout, &output, &circuit, &key_ref)
            } else {
                module_ref.execute_bdd_circuit_1w_to_1w_tmp_bytes::<_, u8, _, _, _>(&bits_layout, &output, &circuit, &key_ref)
            };
            let tb = if two_words {
                module_test.execute_bdd_circuit_2w_to_1w_tmp_bytes::<_, u8, _, _, _>(&bits_layout, &output, &circuit, &key_test)
            } else {
                module_test.execute_bdd_circuit_1w_to_1w_tmp_bytes::<_, u8, _, _, _>(&bits_layout, &output, &circuit, &key_test)
            };
            with_scratch::<BR, _>(rb, |s| {
                if two_words {
                    module_ref.execute_bdd_circuit_2w_to_1w(&mut expected, &res_ref, &right_ref, &circuit, &key_ref, s)
                } else {
                    module_ref.execute_bdd_circuit_1w_to_1w(&mut expected, &res_ref, &circuit, &key_ref, s)
                }
            });
            with_scratch::<BT, _>(tb, |s| {
                if two_words {
                    module_test.execute_bdd_circuit_2w_to_1w(&mut actual, &res_test, &right_test, &circuit, &key_test, s)
                } else {
                    module_test.execute_bdd_circuit_1w_to_1w(&mut actual, &res_test, &circuit, &key_test, s)
                }
            });
            assert_eq!(snapshot_glwe::<BR, _>(&expected), snapshot_glwe::<BT, _>(&actual));
            for threads in [1usize, 2, 4] {
                let bytes = if two_words {
                    module_test.execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes::<_, u8, _, _, _>(
                        threads,
                        &bits_layout,
                        &output,
                        &circuit,
                        &key_test,
                    )
                } else {
                    module_test.execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes::<_, u8, _, _, _>(
                        threads,
                        &bits_layout,
                        &output,
                        &circuit,
                        &key_test,
                    )
                };
                with_scratch::<BT, _>(bytes, |s| {
                    if two_words {
                        module_test.execute_bdd_circuit_2w_to_1w_multi_thread(
                            threads,
                            &mut actual,
                            &res_test,
                            &right_test,
                            &circuit,
                            &key_test,
                            s,
                        )
                    } else {
                        module_test.execute_bdd_circuit_1w_to_1w_multi_thread(
                            threads,
                            &mut actual,
                            &res_test,
                            &circuit,
                            &key_test,
                            s,
                        )
                    }
                });
                assert_eq!(snapshot_glwe::<BR, _>(&expected), snapshot_glwe::<BT, _>(&actual));
            }
        }
    }
}

/// Checks randomized lifecycle overrides against the callable algorithm using
/// the *same* backend's selected samplers, so this probe makes no assumption that
/// unrelated backends draw identical ciphertexts from equal seeds.
pub fn test_lifecycle_reference<BE: ParityBackend>(module: &Module<BE>)
where
    Module<BE>: BDDKeyEncryptSk<CGGI, BE>
        + CircuitBootstrappingKeyEncryptSk<CGGI, BE>
        + BlindRotationKeyEncryptSk<CGGI, BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEToLWEKeyEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + FheUintPreparedEncryptSk<u8, BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEExternalProduct<BE>,
{
    for bridge in [false, true] {
        let layout = key_layout(module.n(), bridge);
        let mut sk = module.glwe_secret_alloc(layout.cbt_layout.brk_layout.rank);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([83; 32]));
        let mut sk_lwe = module.lwe_secret_alloc(6usize.into());
        module.lwe_secret_fill_binary_block(&mut sk_lwe, 3, &mut Source::new([89; 32]));

        let mut expected = BDDKey::alloc_from_infos(module, &layout);
        let mut actual = BDDKey::alloc_from_infos(module, &layout);
        let circuit_bytes =
            crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_tmp_bytes_reference::<_, _, CGGI, BE>(
                module,
                &layout.cbt_layout,
            );
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(circuit_bytes, |scratch| {
            crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_reference::<_, _, _, CGGI, BE>(
                module,
                &mut expected.cbt,
                &sk_lwe,
                &sk,
                &mut Source::new([109; 32]),
                &mut Source::new([113; 32]),
                scratch,
            );
        });
        with_scratch::<BE, _>(
            module.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&layout.cbt_layout),
            |scratch| {
                module.circuit_bootstrapping_key_encrypt_sk(
                    &mut actual.cbt,
                    &sk_lwe,
                    &sk,
                    &mut Source::new([109; 32]),
                    &mut Source::new([113; 32]),
                    scratch,
                );
            },
        );
        assert!(
            raw_key_bytes(module, &expected, &layout) == raw_key_bytes(module, &actual, &layout),
            "key encryption parity mismatch"
        );
        let reference_bytes = crate::reference::bdd::bdd_key_encrypt_sk_tmp_bytes_reference::<CGGI, BE, _>(module, &layout);
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(reference_bytes, |s| {
            crate::reference::bdd::bdd_key_encrypt_sk_reference::<CGGI, BE, _, _>(
                module,
                &mut expected,
                &sk_lwe,
                &sk,
                &mut Source::new([97; 32]),
                &mut Source::new([101; 32]),
                s,
            )
        });
        with_scratch::<BE, _>(module.bdd_key_encrypt_sk_tmp_bytes(&layout), |s| {
            module.bdd_key_encrypt_sk(
                &mut actual,
                &sk_lwe,
                &sk,
                &mut Source::new([97; 32]),
                &mut Source::new([101; 32]),
                s,
            )
        });
        assert!(
            raw_key_bytes(module, &expected, &layout) == raw_key_bytes(module, &actual, &layout),
            "key encryption parity mismatch"
        );
        let mut sk_prepared = module.glwe_secret_prepared_alloc(layout.cbt_layout.brk_layout.rank);
        module.glwe_secret_prepare(&mut sk_prepared, &sk);
        let output = GGSWLayout {
            n: module.n().into(),
            base2k: 15usize.into(),
            dnum: 2usize.into(),
            k_digit: 15usize.into(),
            k_aux: (15 + module.n().ilog2() as usize).into(),
            rank: layout.cbt_layout.brk_layout.rank,
        };

        for value in [0u8, 0xA5, u8::MAX] {
            let mut expected = module.alloc_fhe_uint_prepared_from_infos(&output);
            let mut actual = module.alloc_fhe_uint_prepared_from_infos(&output);
            let bytes = crate::reference::bdd::fhe_uint_prepared_encrypt_sk_tmp_bytes_reference::<BE, _>(module, &output);
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(bytes, |s| {
                crate::reference::bdd::fhe_uint_prepared_encrypt_sk_reference::<u8, BE, _>(
                    module,
                    &mut expected,
                    value,
                    &sk_prepared,
                    &mut Source::new([103; 32]),
                    &mut Source::new([107; 32]),
                    s,
                )
            });
            with_scratch::<BE, _>(
                <Module<BE> as FheUintPreparedEncryptSk<u8, BE>>::fhe_uint_prepared_encrypt_sk_tmp_bytes(module, &output),
                |s| {
                    module.fhe_uint_prepared_encrypt_sk(
                        &mut actual,
                        value,
                        &sk_prepared,
                        &mut Source::new([103; 32]),
                        &mut Source::new([107; 32]),
                        s,
                    )
                },
            );
            for (expected_bit, actual_bit) in expected.bits.iter().zip(&actual.bits) {
                let expected_noise = expected_bit.noise().expect("reference bit is missing metadata");
                let actual_noise = actual_bit.noise().expect("delegated bit is missing metadata");
                assert_eq!(expected_noise, actual_noise);
                assert_eq!(expected_noise.precision(), expected_bit.k());
                assert_eq!(actual_noise.precision(), actual_bit.k());
            }
            assert_eq!(observe(module, &expected), observe(module, &actual));
        }
    }
}
