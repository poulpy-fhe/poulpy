//! Circuit-bootstrap parity over transferred coefficient-domain inputs and keys.
use super::{ParityBackend, fixture_ggsw, snapshot_ggsw, with_scratch};
use crate::{
    api::{CircuitBootstrappingExecute, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPreparedFactory},
    blind_rotation::{BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyLayout,
    },
};
use poulpy_core::{
    EncryptionLayout, LWEEncryptSk, TransferInto,
    layouts::{
        Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GLWEAutomorphismKeyLayout, GLWESecretSampling, LWELayout, LWESecretSampling,
        ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    layouts::{HostBytesBackend, Module},
    source::Source,
};

/// Compares direct and reusable-plan constant/exponent execution, including
/// equal-gap and repacking paths, different key radices, and extended domains.
/// Both backends prepare the same transferred raw key independently. Each call
/// receives exactly its own advertised scratch budget, poisoned and guarded.
/// No host-readable view is required from either backend.
pub fn test_circuit_bootstrapping_parity<BR: ParityBackend, BT: ParityBackend>(reference: &Module<BR>, tested: &Module<BT>)
where
    Module<BR>: CircuitBootstrappingExecute<CGGI, BR>
        + CircuitBootstrappingKeyEncryptSk<CGGI, BR>
        + CircuitBootstrappingKeyPreparedFactory<CGGI, BR>
        + GLWESecretSampling<BR>
        + LWESecretSampling<BR>
        + LWEEncryptSk<BR>,
    Module<BT>: CircuitBootstrappingExecute<CGGI, BT> + CircuitBootstrappingKeyPreparedFactory<CGGI, BT>,
{
    assert_eq!(reference.n(), tested.n());
    let n = reference.n();
    assert!(n >= 32);
    let host = Module::<HostBytesBackend>::new(n as u64);
    for (rank, block_size) in [(1usize, 1usize), (2, 3)] {
        let layout = CircuitBootstrappingKeyLayout {
            brk_layout: BlindRotationKeyLayout {
                n_glwe: n.into(),
                n_lwe: 6usize.into(),
                base2k: 13usize.into(),
                dnum: 3usize.into(),
                k_aux: (13 + n.ilog2() as usize).into(),
                rank: rank.into(),
            },
            atk_layout: GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: 11usize.into(),
                dnum: 4usize.into(),
                k_aux: (11 + n.ilog2() as usize).into(),
                dsize: Dsize(1),
                rank: rank.into(),
            },
            tsk_layout: GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: 12usize.into(),
                dnum: 4usize.into(),
                k_aux: (12 + n.ilog2() as usize).into(),
                dsize: Dsize(1),
                rank: rank.into(),
            },
        };
        let output = GGSWLayout {
            n: n.into(),
            base2k: 15usize.into(),
            dnum: 2usize.into(),
            dsize: Dsize(1),
            k_aux: (15 + n.ilog2() as usize).into(),
            rank: rank.into(),
        };
        let mut source = Source::new([11; 32]);
        let mut sk_glwe = reference.glwe_secret_alloc(rank.into());
        reference.glwe_secret_fill_ternary_prob(&mut sk_glwe, 0.5, &mut source);
        let mut sk_lwe = reference.lwe_secret_alloc(6usize.into());
        reference.lwe_secret_fill_binary_block(&mut sk_lwe, block_size, &mut source);
        let mut raw_ref = CircuitBootstrappingKey::alloc_from_infos(reference, &layout);
        let enc = CircuitBootstrappingEncryptionInfos::from_default_sigma(&layout).unwrap();
        with_scratch::<BR, _>(reference.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&layout), |scratch| {
            reference.circuit_bootstrapping_key_encrypt_sk(
                &mut raw_ref,
                &sk_lwe,
                &sk_glwe,
                &enc,
                &mut Source::new([19; 32]),
                &mut Source::new([23; 32]),
                scratch,
            );
        });
        let mut raw_test = CircuitBootstrappingKey::alloc_from_infos(tested, &layout);
        raw_ref.transfer_into(&mut raw_test);
        assert_eq!(raw_ref.block_size(), raw_test.block_size());
        assert_eq!(raw_ref.brk_infos(), raw_test.brk_infos());
        assert_eq!(raw_ref.atk_infos(), raw_test.atk_infos());
        assert_eq!(raw_ref.tsk_infos(), raw_test.tsk_infos());
        let mut key_ref = reference.circuit_bootstrapping_key_prepared_alloc_from_infos(&layout);
        let mut key_test = tested.circuit_bootstrapping_key_prepared_alloc_from_infos(&layout);
        with_scratch::<BR, _>(reference.circuit_bootstrapping_key_prepare_tmp_bytes(&layout), |s| {
            reference.circuit_bootstrapping_key_prepare(&mut key_ref, &raw_ref, s)
        });
        with_scratch::<BT, _>(tested.circuit_bootstrapping_key_prepare_tmp_bytes(&layout), |s| {
            tested.circuit_bootstrapping_key_prepare(&mut key_test, &raw_test, s)
        });
        assert_eq!(key_ref.block_size(), block_size);
        assert_eq!(key_ref.block_size(), key_test.block_size());
        assert_eq!(key_ref.brk_infos(), key_test.brk_infos());
        assert_eq!(key_ref.atk_infos(), key_test.atk_infos());
        assert_eq!(key_ref.tsk_infos(), key_test.tsk_infos());
        let lwe_infos = EncryptionLayout::new_from_default_sigma(LWELayout {
            n: 6usize.into(),
            base2k: 14usize.into(),
            k: 28usize.into(),
        })
        .unwrap();
        let mut plain_host = host.lwe_plaintext_alloc(14usize.into(), 3usize.into());
        plain_host.encode_i64(1, 2usize.into());
        let mut plain = reference.lwe_plaintext_alloc(14usize.into(), 3usize.into());
        plain_host.transfer_into(&mut plain);
        let mut input_ref = reference.lwe_alloc_from_infos(&lwe_infos);
        with_scratch::<BR, _>(reference.lwe_encrypt_sk_tmp_bytes(&lwe_infos), |s| {
            reference.lwe_encrypt_sk(
                &mut input_ref,
                &plain,
                &sk_lwe,
                &lwe_infos,
                &mut Source::new([29; 32]),
                &mut Source::new([31; 32]),
                s,
            )
        });
        let mut input_test = tested.lwe_alloc_from_infos(&lwe_infos);
        input_ref.transfer_into(&mut input_test);
        for extension in [0, 3] {
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                key_test.prepare_to_constant(tested, &output, 1, extension)
            }));
            assert!(rejected.is_err(), "plan accepted a non-power-of-two extension");
        }
        for extension in [1, 2] {
            for gap in [None, Some(n.ilog2() as usize - 1), Some(1)] {
                let plan_ref = match gap {
                    None => key_ref.prepare_to_constant(reference, &output, 1, extension),
                    Some(gap) => key_ref.prepare_to_exponent(reference, gap, &output, 1, extension),
                };
                let plan_test = match gap {
                    None => key_test.prepare_to_constant(tested, &output, 1, extension),
                    Some(gap) => key_test.prepare_to_exponent(tested, gap, &output, 1, extension),
                };
                assert_eq!(plan_ref.output_layout(), output);
                assert_eq!(plan_ref.output_layout(), plan_test.output_layout());
                let mut invalid_layout = output;
                invalid_layout.dnum = 1usize.into();
                let mut invalid = fixture_ggsw(tested, &invalid_layout, 47);
                let before = snapshot_ggsw::<BT, _>(&invalid);
                let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_scratch::<BT, _>(plan_test.execute_tmp_bytes(tested, &key_test), |s| {
                        plan_test.execute(tested, &mut invalid, &input_test, &key_test, s)
                    });
                }));
                assert!(failure.is_err(), "prepared plan accepted incompatible output layout");
                assert_eq!(snapshot_ggsw::<BT, _>(&invalid), before, "invalid output was modified");
                let mut expected = fixture_ggsw(reference, &output, 37);
                let mut actual = fixture_ggsw(tested, &output, 37);
                with_scratch::<BR, _>(plan_ref.execute_tmp_bytes(reference, &key_ref), |s| {
                    plan_ref.execute(reference, &mut expected, &input_ref, &key_ref, s)
                });
                with_scratch::<BT, _>(plan_test.execute_tmp_bytes(tested, &key_test), |s| {
                    plan_test.execute(tested, &mut actual, &input_test, &key_test, s)
                });
                assert_eq!(snapshot_ggsw::<BR, _>(&expected), snapshot_ggsw::<BT, _>(&actual));
                // Reuse the same prepared plan after replacing every output coefficient.
                actual = fixture_ggsw(tested, &output, 41);
                with_scratch::<BT, _>(plan_test.execute_tmp_bytes(tested, &key_test), |s| {
                    plan_test.execute(tested, &mut actual, &input_test, &key_test, s)
                });
                assert_eq!(snapshot_ggsw::<BR, _>(&expected), snapshot_ggsw::<BT, _>(&actual));
                for legacy in [false, true] {
                    let bytes = match gap {
                        None if legacy => {
                            tested.circuit_bootstrapping_execute_tmp_bytes(block_size, extension, &output, &key_test)
                        }
                        None => {
                            tested.circuit_bootstrapping_execute_to_constant_tmp_bytes(block_size, extension, &output, &key_test)
                        }
                        Some(gap) => tested.circuit_bootstrapping_execute_to_exponent_tmp_bytes(
                            gap, 1, block_size, extension, &output, &key_test,
                        ),
                    };
                    actual = fixture_ggsw(tested, &output, 43);
                    with_scratch::<BT, _>(bytes, |s| match gap {
                        None => key_test.execute_to_constant(tested, &mut actual, &input_test, 1, extension, s),
                        Some(gap) => key_test.execute_to_exponent(tested, gap, &mut actual, &input_test, 1, extension, s),
                    });
                    assert_eq!(snapshot_ggsw::<BR, _>(&expected), snapshot_ggsw::<BT, _>(&actual));
                    if gap.is_some() {
                        break;
                    }
                }
            }
        }
    }
}
