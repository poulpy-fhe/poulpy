use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps, CKKSEvalModOps, CKKSPolynomialEvaluationOps,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType,
        EncapsulationKeysLayout, EncodedLut, SparseSecretEncapsulation, eval_mod::EvalModPlan,
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw, precision_stats,
        },
        reference_encoder::ReferenceEncoder,
    },
};

const K_INTERVAL: usize = 16;
const LOG_INTERVAL_REDUCTION: usize = 3;
const EXP_DEGREE: usize = 31;
const INPUT_LOG_DELTA: usize = 40;
const EVAL_LOG_DELTA: usize = 50;
const LUT_LOG_DELTA: usize = 55;

#[derive(Clone, Copy)]
enum Case {
    General,
    Multi,
    Binary,
}

pub fn test_functional_bootstrapping_e2e<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSPolynomialEvaluationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    run_case::<BE, F, E>(Case::General, params, module, host_module);
}

pub fn test_functional_bootstrapping_multi_e2e<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSPolynomialEvaluationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    run_case::<BE, F, E>(Case::Multi, params, module, host_module);
}

pub fn test_functional_bootstrapping_binary_e2e<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSPolynomialEvaluationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    run_case::<BE, F, E>(Case::Binary, params, module, host_module);
}

fn run_case<BE, F, E>(case: Case, params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSPolynomialEvaluationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let table_values: &[&[usize]] = match case {
        Case::General => &[&[0, 1, 0, 0]],
        Case::Multi => &[
            &[5, 2, 7, 0, 3, 6, 1, 4],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        ],
        Case::Binary => &[&[3, 1]],
    };
    let tables: Vec<Vec<F>> = table_values
        .iter()
        .map(|table| table.iter().map(|&value| F::from_usize(value).unwrap()).collect())
        .collect();
    let p = tables[0].len();
    let plan = fbt_plan(params.base2k);
    let coeffs_meta = CoeffsMeta::from_delta_budget(LUT_LOG_DELTA, params.base2k);

    let host_luts: Vec<EncodedLut<CKKSPlaintextOwned<HostBytesBackend>>> = match case {
        Case::Binary => vec![
            EncodedLut::binary(
                host_module,
                tables[0][0],
                tables[0][1],
                EXP_DEGREE,
                K_INTERVAL,
                LOG_INTERVAL_REDUCTION,
                params.base2k.into(),
                coeffs_meta,
                SplitStrategy::MinDepth,
            )
            .unwrap(),
        ],
        Case::General | Case::Multi => tables
            .iter()
            .enumerate()
            .map(|(index, table)| {
                // A later LUT with a lower coefficient scale retains more
                // output width than the LUT evaluated before it.
                let coeffs_meta = if index == 1 {
                    CoeffsMeta::from_delta_budget(LUT_LOG_DELTA - 5, params.base2k)
                } else {
                    coeffs_meta
                };
                EncodedLut::general(host_module, table, params.base2k.into(), coeffs_meta, SplitStrategy::MinDepth).unwrap()
            })
            .collect(),
    };
    let log_msg_ratio = host_luts[0].log_msg_ratio();
    let log_modulus_in = INPUT_LOG_DELTA + log_msg_ratio;
    let k_in = plan.input_k(log_modulus_in);
    let output_k = log_modulus_in + 2 * INPUT_LOG_DELTA;
    let functional_k = plan.functional_bootstrap_k(output_k, INPUT_LOG_DELTA, &host_luts[0]).unwrap();
    let expected_functional_k = match case {
        Case::General => 769,
        Case::Multi => 814,
        Case::Binary => 668,
    };
    assert_eq!(functional_k, expected_functional_k);
    let k_boot = functional_k.next_multiple_of(2 * params.base2k);
    let backend_luts: Vec<_> = host_luts.iter().map(|lut| lut.to_backend(module)).collect();
    let tp = CKKSTestParams {
        k: k_boot,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: INPUT_LOG_DELTA,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 10,
        hw: 192,
        dsize: 2,
        rank: 1,
        ..params
    };
    let encoder = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();

    let initial_tmp;
    let mut scratch = {
        let mut ct = module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into());
        ct.set_meta(tp.prec().meta);
        initial_tmp = module.ckks_all_ops_with_atk_tmp_bytes(
            &ct,
            &tp.tsk_layout(),
            &tp.atk_layout(),
            &ckks_spec(params.n, params.base2k, EVAL_LOG_DELTA, params.base2k),
        );
        ScratchOwned::<BE>::alloc(initial_tmp)
    };
    let ctx = BootstrappingContext::<BE, F>::compile(module, params.base2k.into(), &plan, &mut scratch.borrow()).unwrap();
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let output_spec = ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, k_boot - INPUT_LOG_DELTA);
    let input_spec = ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, k_in - INPUT_LOG_DELTA);
    let boot_tmp = module.ckks_functional_bootstrap_tmp_bytes(&output_spec, &input_spec, &ctx, &backend_luts, &keys_layout);
    if backend_luts.iter().any(|lut| lut.requires_eval_mod()) {
        let boot_layout = crate::CKKSLayout {
            glwe_layout: output_spec.glwe_layout,
            meta: CKKSMeta::default(),
        };
        let boot_ct_bytes = module.glwe_bytes_of_from_infos(&boot_layout);
        let eval_mod_tmp = module.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), &keys_layout.tensor_key);
        assert!(boot_tmp >= 4 * boot_ct_bytes + eval_mod_tmp);
    }
    if boot_tmp > initial_tmp {
        scratch = ScratchOwned::<BE>::alloc(boot_tmp);
    }

    let (sk_raw, sk) = gen_sk_with_raw(&tp, module, host_module, [0u8; 32]);
    let (mut xs, mut xe, mut xa) = (Source::new([7u8; 32]), Source::new([2u8; 32]), Source::new([1u8; 32]));
    let keys = ctx
        .generate_keys(
            module,
            &sk_raw,
            &keys_layout,
            &mut xs,
            &mut xe,
            &mut xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(module, &mut scratch.borrow());

    let mut op_scratch = ScratchOwned::<BE>::alloc(boot_tmp);

    let mut source = Source::new([9u8; 32]);
    let sample = |source: &mut Source| ((source.next_f64(0.0, 1.0) * p as f64) as usize).min(p - 1);
    for slots_kind in [SlotsKind::Complex, SlotsKind::Real] {
        let messages_re: Vec<_> = (0..params.n / 2).map(|_| sample(&mut source)).collect();
        let messages_im: Vec<_> = (0..params.n / 2)
            .map(|_| {
                if slots_kind == SlotsKind::Real {
                    0
                } else {
                    sample(&mut source)
                }
            })
            .collect();
        let re: Vec<F> = messages_re.iter().map(|&value| F::from_usize(value).unwrap()).collect();
        let im: Vec<F> = messages_im.iter().map(|&value| F::from_usize(value).unwrap()).collect();
        let mut ct = ckks_encrypt_with_prec(
            &tp,
            module,
            host_module,
            &encoder,
            &sk,
            k_in,
            &re,
            &im,
            input_spec,
            &mut scratch.borrow(),
        );
        ct.set_slots(slots_kind);

        // One entry point for one LUT or many; the slot kind of the input
        // selects the pipeline.
        let mut outputs: Vec<_> = backend_luts
            .iter()
            .map(|_| module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into()))
            .collect();
        module
            .ckks_functional_bootstrap(&mut outputs, &ct, &ctx, &backend_luts, &keys, &mut op_scratch.borrow())
            .unwrap();

        for (index, (output, table)) in outputs.iter().zip(&tables).enumerate() {
            let lut = &backend_luts[index];
            let output_log_delta = INPUT_LOG_DELTA + lut.log_msg_ratio();
            let eval_mod_bits = if lut.requires_eval_mod() {
                plan.eval_mod().consumed_bits()
            } else {
                0
            };
            let expected_k =
                k_boot - plan.coeffs_to_slots().consumed_bits() - eval_mod_bits - lut.consumed_bits(output_log_delta);
            assert_eq!(output.k().as_usize(), expected_k);
            assert_eq!(output.log_delta(), output_log_delta);
            assert_eq!(output.slots(), slots_kind);
            let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(&tp, module, &encoder, output, &sk, &mut scratch.borrow());
            let want_re: Vec<F> = messages_re.iter().map(|&message| table[message]).collect();
            let want_im: Vec<F> = if slots_kind == SlotsKind::Real {
                vec![F::zero(); messages_im.len()]
            } else {
                messages_im.iter().map(|&message| table[message]).collect()
            };
            for (got, want, part) in [(&got_re, &want_re, "re"), (&got_im, &want_im, "im")] {
                let stats = precision_stats(got, want, output.log_delta());
                assert!(
                    stats.min_log2_prec >= 8.0,
                    "functional bootstrapping [{index}] ({part}) worst slot had {:.1} bits (average {:.1})",
                    stats.min_log2_prec,
                    stats.avg_log2_prec
                );
            }
        }

        if slots_kind == SlotsKind::Complex && matches!(case, Case::General) {
            let wrong_ratio_lut = EncodedLut::binary(
                host_module,
                F::zero(),
                F::from_usize(1).unwrap(),
                EXP_DEGREE,
                K_INTERVAL,
                LOG_INTERVAL_REDUCTION,
                params.base2k.into(),
                coeffs_meta,
                SplitStrategy::MinDepth,
            )
            .unwrap();
            let mut output = module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into());
            let error = module
                .ckks_functional_bootstrap(
                    std::slice::from_mut(&mut output),
                    &ct,
                    &ctx,
                    std::slice::from_ref(&wrong_ratio_lut),
                    &keys,
                    &mut op_scratch.borrow(),
                )
                .unwrap_err();
            assert!(error.to_string().contains("log_msg_ratio"));

            // Reusing a wider allocation at a narrower effective width must
            // remain within the scratch size queried for that effective width.
            let mut wide_output = module.ckks_ciphertext_alloc(params.base2k.into(), (k_boot + 4 * params.base2k).into());
            wide_output.set_k(k_boot.into());
            module
                .ckks_functional_bootstrap(
                    std::slice::from_mut(&mut wide_output),
                    &ct,
                    &ctx,
                    &backend_luts,
                    &keys,
                    &mut op_scratch.borrow(),
                )
                .unwrap();
        }

        if slots_kind == SlotsKind::Complex && matches!(case, Case::Multi) {
            let luts = &backend_luts;
            let mut outputs: Vec<_> = luts
                .iter()
                .map(|_| module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into()))
                .collect();
            outputs[1].set_k((k_boot - params.base2k).into());
            let error = module
                .ckks_functional_bootstrap(&mut outputs, &ct, &ctx, luts, &keys, &mut op_scratch.borrow())
                .unwrap_err();
            assert!(error.to_string().contains("share one rank-1 layout"));

            outputs[1].set_k(k_boot.into());
            let mixed_ratio_luts = vec![
                EncodedLut::general(
                    host_module,
                    &tables[0],
                    params.base2k.into(),
                    coeffs_meta,
                    SplitStrategy::MinDepth,
                )
                .unwrap(),
                EncodedLut::general(
                    host_module,
                    &tables[0][..4],
                    params.base2k.into(),
                    coeffs_meta,
                    SplitStrategy::MinDepth,
                )
                .unwrap(),
            ];
            let error = module
                .ckks_functional_bootstrap(
                    &mut outputs[..2],
                    &ct,
                    &ctx,
                    &mixed_ratio_luts,
                    &keys,
                    &mut op_scratch.borrow(),
                )
                .unwrap_err();
            assert!(error.to_string().contains("same message ratio"));

            let binary_luts = vec![
                EncodedLut::general(
                    host_module,
                    &tables[0][..2],
                    params.base2k.into(),
                    coeffs_meta,
                    SplitStrategy::MinDepth,
                )
                .unwrap(),
                EncodedLut::binary(
                    host_module,
                    tables[0][0],
                    tables[0][1],
                    EXP_DEGREE,
                    K_INTERVAL,
                    LOG_INTERVAL_REDUCTION,
                    params.base2k.into(),
                    coeffs_meta,
                    SplitStrategy::MinDepth,
                )
                .unwrap(),
            ];
            let error = module
                .ckks_functional_bootstrap(&mut outputs[..2], &ct, &ctx, &binary_luts, &keys, &mut op_scratch.borrow())
                .unwrap_err();
            assert!(error.to_string().contains("log_msg_ratio"));
        }
    }

    if matches!(case, Case::General) {
        let insufficient_k = INPUT_LOG_DELTA + plan.pre_mod_up_consumed_bits() - 1;
        let zeros = vec![F::zero(); params.n / 2];
        let ct = ckks_encrypt_with_prec(
            &tp,
            module,
            host_module,
            &encoder,
            &sk,
            insufficient_k,
            &zeros,
            &zeros,
            ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, insufficient_k - INPUT_LOG_DELTA),
            &mut scratch.borrow(),
        );
        let mut output = module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into());
        let error = module
            .ckks_functional_bootstrap(
                std::slice::from_mut(&mut output),
                &ct,
                &ctx,
                &backend_luts,
                &keys,
                &mut op_scratch.borrow(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("functional bootstrap needs log_budget"));
    }
}

fn fbt_plan(base2k: usize) -> BootstrappingPlan {
    let slots_to_coeffs = DFTPlan::new(
        DFTType::Decode,
        vec![(2, 4), (3, 4), (2, 4)],
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(45, 2),
    )
    .unwrap()
    .with_scaling(0.5)
    .unwrap();
    let coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        vec![(2, 4), (3, 4), (2, 4)],
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(50, 2),
    )
    .unwrap()
    .with_scaling(1.0 / K_INTERVAL as f64)
    .unwrap();
    BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 32 }),
            eval_round_plus: None,
        },
        coeffs_to_slots,
        EvalModPlan::complex_exponential(
            EXP_DEGREE,
            K_INTERVAL,
            LOG_INTERVAL_REDUCTION,
            SplitStrategy::MinDepth,
            CoeffsMeta::from_delta_budget(EVAL_LOG_DELTA, base2k),
            EVAL_LOG_DELTA,
        ),
        slots_to_coeffs,
    )
    .unwrap()
}
