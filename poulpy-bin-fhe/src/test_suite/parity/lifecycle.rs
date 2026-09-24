//! Key lifecycle and packed-to-prepared integer parity.
use super::{ParityBackend, fixture_glwe, snapshot_glwe, with_scratch};
use crate::{
    api::{
        BDDKeyEncryptSk, BDDKeyPreparedFactory, BlindRotationKeyEncryptSk, CircuitBootstrappingKeyEncryptSk,
        ExecuteBDDCircuit1WTo1W, ExecuteBDDCircuit2WTo1W, FheUintPrepare, FheUintPreparedEncryptSk,
    },
    bdd_arithmetic::{
        BDDEncryptionInfos, BDDKey, BDDKeyInfos, BDDKeyLayout, FheUint, FheUintPrepared, FheUintPreparedFactory,
        GetBitCircuitInfo, Node,
    },
    blind_rotation::{BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};
use poulpy_core::{
    GGLWEToGGSWKeyEncryptSk, GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEExternalProduct, GLWESwitchingKeyEncryptSk,
    GLWEToLWESwitchingKeyEncryptSk, TransferInto,
    layouts::{
        Dsize, GGLWEToGGSWKeyLayout, GGSWInfos, GGSWLayout, GGSWPreparedFactory, GGSWPreparedToBackendRef,
        GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWESecretPreparedFactory, GLWESecretSampling, GLWESwitchingKeyLayout,
        GLWEToLWEKeyLayout, LWEInfos, LWESecretSampling, ModuleCoreAlloc,
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
                dsize: Dsize(1),
                k_aux: (11 + aux).into(),
                rank: rank.into(),
            },
            tsk_layout: GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: 12usize.into(),
                dnum: 4usize.into(),
                dsize: Dsize(1),
                k_aux: (12 + aux).into(),
                rank: rank.into(),
            },
        },
        ks_glwe_layout: bridge.then_some(GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: 12usize.into(),
            dnum: 4usize.into(),
            dsize: Dsize(1),
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

fn raw_key_bytes<B: ParityBackend>(module: &Module<B>, key: &BDDKey<B::OwnedBuf, CGGI, i64>, layout: &BDDKeyLayout) -> Vec<u8> {
    let host = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut copy = BDDKey::alloc_from_infos(&host, layout);
    key.transfer_into(&mut copy);
    let mut bytes = Vec::new();
    copy.write_to(&mut bytes).unwrap();
    bytes
}

fn observe<B: ParityBackend>(
    module: &Module<B>,
    value: &FheUintPrepared<B::OwnedBuf, u8, B>,
) -> (GGSWLayout, Vec<super::GlweSnapshot>)
where
    Module<B>: GLWEExternalProduct<B>,
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
                with_scratch::<B, _>(module.glwe_external_product_tmp_bytes(&output, &input, bit), |s| {
                    module.glwe_external_product(&mut output, &input, &bit.to_backend_ref(), s)
                });
                snapshot_glwe::<B, _>(&output)
            })
            .collect(),
    )
}

/// Transfers a single raw BDD key and packed input, prepares independently, and
/// checks every prepared bit through a coefficient-domain external product.
/// Covers optional rank switching, full/custom ranges, empty ranges, reuse, and
/// supported worker counts. Prepared representations are never compared as bytes.
pub fn test_lifecycle_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: BDDKeyEncryptSk<CGGI, BR>
        + BDDKeyPreparedFactory<CGGI, BR>
        + FheUintPrepare<CGGI, BR>
        + GGSWPreparedFactory<BR>
        + GLWEExternalProduct<BR>
        + GLWESecretSampling<BR>
        + LWESecretSampling<BR>
        + ExecuteBDDCircuit1WTo1W<BR>
        + ExecuteBDDCircuit2WTo1W<BR>,
    Module<BT>: BDDKeyPreparedFactory<CGGI, BT>
        + FheUintPrepare<CGGI, BT>
        + GGSWPreparedFactory<BT>
        + GLWEExternalProduct<BT>
        + ExecuteBDDCircuit1WTo1W<BT>
        + ExecuteBDDCircuit2WTo1W<BT>,
{
    assert_eq!(reference.n(), tested.n());
    for bridge in [false, true] {
        let layout = key_layout(reference.n(), bridge);
        let rank = layout.cbt_layout.brk_layout.rank;
        let output = GGSWLayout {
            n: reference.n().into(),
            base2k: 15usize.into(),
            dnum: 2usize.into(),
            dsize: Dsize(1),
            k_aux: (15 + reference.n().ilog2() as usize).into(),
            rank,
        };
        let mut sk = reference.glwe_secret_alloc(rank);
        reference.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([61; 32]));
        let mut sk_lwe = reference.lwe_secret_alloc(6usize.into());
        reference.lwe_secret_fill_binary_block(&mut sk_lwe, 3, &mut Source::new([67; 32]));
        let mut raw_ref = BDDKey::alloc_from_infos(reference, &layout);
        let enc = BDDEncryptionInfos::from_default_sigma(&layout).unwrap();
        with_scratch::<BR, _>(reference.bdd_key_encrypt_sk_tmp_bytes(&layout), |s| {
            reference.bdd_key_encrypt_sk(
                &mut raw_ref,
                &sk_lwe,
                &sk,
                &enc,
                &mut Source::new([71; 32]),
                &mut Source::new([73; 32]),
                s,
            )
        });
        let mut raw_test = BDDKey::alloc_from_infos(tested, &layout);
        raw_ref.transfer_into(&mut raw_test);
        assert!(
            raw_key_bytes(reference, &raw_ref, &layout) == raw_key_bytes(tested, &raw_test, &layout),
            "raw key transfer mismatch"
        );
        let mut key_ref = reference.alloc_bdd_key_from_infos(&layout);
        let mut key_test = tested.alloc_bdd_key_from_infos(&layout);
        with_scratch::<BR, _>(reference.prepare_bdd_key_tmp_bytes(&layout), |s| {
            reference.prepare_bdd_key(&mut key_ref, &raw_ref, s)
        });
        with_scratch::<BT, _>(tested.prepare_bdd_key_tmp_bytes(&layout), |s| {
            tested.prepare_bdd_key(&mut key_test, &raw_test, s)
        });
        assert_eq!(key_ref.cbt_infos().brk_layout, key_test.cbt_infos().brk_layout);
        assert_eq!(key_ref.ks_lwe_infos(), key_test.ks_lwe_infos());
        assert_eq!(key_ref.ks_glwe_infos(), key_test.ks_glwe_infos());
        let bits_layout = GLWELayout {
            n: reference.n().into(),
            base2k: 14usize.into(),
            k: 31usize.into(),
            rank,
        };
        let input_ref = FheUint::<_, u8, i64> {
            bits: fixture_glwe(reference, &bits_layout, 79),
            _phantom: std::marker::PhantomData,
        };
        let mut input_test = FheUint::<_, u8, i64>::alloc_from_infos(tested, &bits_layout);
        input_ref.transfer_into(&mut input_test);
        let mut res_ref = reference.alloc_fhe_uint_prepared_from_infos(&output);
        let mut res_test = tested.alloc_fhe_uint_prepared_from_infos(&output);
        let br_bytes = reference.fhe_uint_prepare_tmp_bytes(3, 1, &output, &bits_layout, &layout);
        let bt_bytes = tested.fhe_uint_prepare_tmp_bytes(3, 1, &output, &bits_layout, &layout);
        with_scratch::<BR, _>(br_bytes, |s| {
            reference.fhe_uint_prepare(&mut res_ref, &input_ref, &key_ref, s)
        });
        with_scratch::<BT, _>(bt_bytes, |s| {
            tested.fhe_uint_prepare(&mut res_test, &input_test, &key_test, s)
        });
        assert_eq!(observe(reference, &res_ref), observe(tested, &res_test));
        for (start, count) in [(1usize, 5usize), (2, 0), (0, 8)] {
            with_scratch::<BR, _>(br_bytes, |s| {
                reference.fhe_uint_prepare_custom(&mut res_ref, &input_ref, start, count, &key_ref, s)
            });
            with_scratch::<BT, _>(bt_bytes, |s| {
                tested.fhe_uint_prepare_custom(&mut res_test, &input_test, start, count, &key_test, s)
            });
            let expected = observe(reference, &res_ref);
            assert_eq!(expected, observe(tested, &res_test));
            for threads in [1usize, 2, 4] {
                let workers = poulpy_hal::execution::worker_count::<BT::TaskExecutor>(threads, count);
                with_scratch::<BT, _>(bt_bytes * workers, |s| {
                    tested.fhe_uint_prepare_custom_multi_thread(threads, &mut res_test, &input_test, start, count, &key_test, s)
                });
                assert_eq!(expected, observe(tested, &res_test));
            }
        }
        let mut right_ref = reference.alloc_fhe_uint_prepared_from_infos(&output);
        let mut right_test = tested.alloc_fhe_uint_prepared_from_infos(&output);
        with_scratch::<BR, _>(br_bytes, |s| {
            reference.fhe_uint_prepare_custom(&mut right_ref, &input_ref, 0, 0, &key_ref, s)
        });
        with_scratch::<BT, _>(bt_bytes, |s| {
            tested.fhe_uint_prepare_custom(&mut right_test, &input_test, 0, 0, &key_test, s)
        });
        for two_words in [false, true] {
            let circuit = SelectBits::new(two_words);
            let mut expected = FheUint::<_, u8, i64>::alloc_from_infos(reference, &bits_layout);
            let mut actual = FheUint::<_, u8, i64>::alloc_from_infos(tested, &bits_layout);
            let rb = if two_words {
                reference.execute_bdd_circuit_2w_to_1w_tmp_bytes::<_, u8, _, _, _>(&circuit, &bits_layout, &output, &key_ref)
            } else {
                reference.execute_bdd_circuit_1w_to_1w_tmp_bytes::<_, u8, _, _, _>(&circuit, &bits_layout, &output, &key_ref)
            };
            let tb = if two_words {
                tested.execute_bdd_circuit_2w_to_1w_tmp_bytes::<_, u8, _, _, _>(&circuit, &bits_layout, &output, &key_test)
            } else {
                tested.execute_bdd_circuit_1w_to_1w_tmp_bytes::<_, u8, _, _, _>(&circuit, &bits_layout, &output, &key_test)
            };
            with_scratch::<BR, _>(rb, |s| {
                if two_words {
                    reference.execute_bdd_circuit_2w_to_1w(&mut expected, &circuit, &res_ref, &right_ref, &key_ref, s)
                } else {
                    reference.execute_bdd_circuit_1w_to_1w(&mut expected, &circuit, &res_ref, &key_ref, s)
                }
            });
            with_scratch::<BT, _>(tb, |s| {
                if two_words {
                    tested.execute_bdd_circuit_2w_to_1w(&mut actual, &circuit, &res_test, &right_test, &key_test, s)
                } else {
                    tested.execute_bdd_circuit_1w_to_1w(&mut actual, &circuit, &res_test, &key_test, s)
                }
            });
            assert_eq!(snapshot_glwe::<BR, _>(&expected), snapshot_glwe::<BT, _>(&actual));
            for threads in [1usize, 2, 4] {
                let bytes = if two_words {
                    tested.execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes::<_, u8, _, _, _>(
                        threads,
                        &circuit,
                        &bits_layout,
                        &output,
                        &key_test,
                    )
                } else {
                    tested.execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes::<_, u8, _, _, _>(
                        threads,
                        &circuit,
                        &bits_layout,
                        &output,
                        &key_test,
                    )
                };
                with_scratch::<BT, _>(bytes, |s| {
                    if two_words {
                        tested.execute_bdd_circuit_2w_to_1w_multi_thread(
                            threads,
                            &mut actual,
                            &circuit,
                            &res_test,
                            &right_test,
                            &key_test,
                            s,
                        )
                    } else {
                        tested.execute_bdd_circuit_1w_to_1w_multi_thread(threads, &mut actual, &circuit, &res_test, &key_test, s)
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
pub fn test_lifecycle_reference<B: ParityBackend>(module: &Module<B>)
where
    Module<B>: BDDKeyEncryptSk<CGGI, B>
        + CircuitBootstrappingKeyEncryptSk<CGGI, B>
        + BlindRotationKeyEncryptSk<CGGI, B>
        + GGLWEToGGSWKeyEncryptSk<B>
        + GLWEAutomorphismKeyEncryptSk<B>
        + GLWEToLWESwitchingKeyEncryptSk<B>
        + GLWESwitchingKeyEncryptSk<B>
        + GLWESecretSampling<B>
        + LWESecretSampling<B>
        + GLWESecretPreparedFactory<B>
        + FheUintPreparedEncryptSk<u8, B>
        + GGSWEncryptSk<B>
        + GGSWPreparedFactory<B>
        + GLWEExternalProduct<B>,
{
    for bridge in [false, true] {
        let layout = key_layout(module.n(), bridge);
        let mut sk = module.glwe_secret_alloc(layout.cbt_layout.brk_layout.rank);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([83; 32]));
        let mut sk_lwe = module.lwe_secret_alloc(6usize.into());
        module.lwe_secret_fill_binary_block(&mut sk_lwe, 3, &mut Source::new([89; 32]));
        let enc = BDDEncryptionInfos::from_default_sigma(&layout).unwrap();
        let mut expected = BDDKey::alloc_from_infos(module, &layout);
        let mut actual = BDDKey::alloc_from_infos(module, &layout);
        let circuit_bytes =
            crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_tmp_bytes_reference::<_, _, CGGI, B>(
                module,
                &layout.cbt_layout,
            );
        with_scratch::<B, _>(circuit_bytes, |scratch| {
            crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_reference::<_, _, _, CGGI, B>(
                module,
                &mut expected.cbt,
                &sk_lwe,
                &sk,
                &enc.cbt,
                &mut Source::new([109; 32]),
                &mut Source::new([113; 32]),
                scratch,
            );
        });
        with_scratch::<B, _>(
            module.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&layout.cbt_layout),
            |scratch| {
                module.circuit_bootstrapping_key_encrypt_sk(
                    &mut actual.cbt,
                    &sk_lwe,
                    &sk,
                    &enc.cbt,
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
        let reference_bytes = crate::reference::bdd::bdd_key_encrypt_sk_tmp_bytes_reference::<CGGI, B, _>(module, &layout);
        with_scratch::<B, _>(reference_bytes, |s| {
            crate::reference::bdd::bdd_key_encrypt_sk_reference::<CGGI, B, _, _>(
                module,
                &mut expected,
                &sk_lwe,
                &sk,
                &enc,
                &mut Source::new([97; 32]),
                &mut Source::new([101; 32]),
                s,
            )
        });
        with_scratch::<B, _>(module.bdd_key_encrypt_sk_tmp_bytes(&layout), |s| {
            module.bdd_key_encrypt_sk(
                &mut actual,
                &sk_lwe,
                &sk,
                &enc,
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
            dsize: Dsize(1),
            k_aux: (15 + module.n().ilog2() as usize).into(),
            rank: layout.cbt_layout.brk_layout.rank,
        };
        let enc = poulpy_core::EncryptionLayout::new_from_default_sigma(output).unwrap();
        for value in [0u8, 0xA5, u8::MAX] {
            let mut expected = module.alloc_fhe_uint_prepared_from_infos(&output);
            let mut actual = module.alloc_fhe_uint_prepared_from_infos(&output);
            let bytes = crate::reference::bdd::fhe_uint_prepared_encrypt_sk_tmp_bytes_reference::<B, _>(module, &output);
            with_scratch::<B, _>(bytes, |s| {
                crate::reference::bdd::fhe_uint_prepared_encrypt_sk_reference::<u8, B, _, _>(
                    module,
                    &mut expected,
                    value,
                    &sk_prepared,
                    &enc,
                    &mut Source::new([103; 32]),
                    &mut Source::new([107; 32]),
                    s,
                )
            });
            with_scratch::<B, _>(
                <Module<B> as FheUintPreparedEncryptSk<u8, B>>::fhe_uint_prepared_encrypt_sk_tmp_bytes(module, &output),
                |s| {
                    module.fhe_uint_prepared_encrypt_sk(
                        &mut actual,
                        value,
                        &sk_prepared,
                        &enc,
                        &mut Source::new([103; 32]),
                        &mut Source::new([107; 32]),
                        s,
                    )
                },
            );
            assert_eq!(observe(module, &expected), observe(module, &actual));
        }
    }
}
