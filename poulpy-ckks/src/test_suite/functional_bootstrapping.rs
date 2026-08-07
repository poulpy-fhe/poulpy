use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps, CKKSPolynomialEvaluationOps},
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType,
        EncapsulationKeysLayout, EncodedLut, SparseSecretEncapsulation, eval_mod::EvalModPlan,
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw, precision_stats, upload_pt,
        },
        reference_encoder::ReferenceEncoder,
    },
};

const K_INTERVAL: usize = 16;
const LOG_INTERVAL_REDUCTION: usize = 3;
const EXP_DEGREE: usize = 31;
const INPUT_LOG_DELTA: usize = 40;
const EVAL_LOG_DELTA: usize = 50;

#[derive(Clone, Copy)]
enum Case {
    General,
    Multi,
    Binary,
}

enum HostLuts {
    Encoded(EncodedLut<CKKSPlaintext<Vec<u8>>>),
    Multi(Vec<EncodedLut<CKKSPlaintext<Vec<u8>>>>),
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
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
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
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
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
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
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
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let table_values: &[&[usize]] = match case {
        Case::General => &[&[0, 1, 0, 0]],
        Case::Multi => &[
            &[5, 2, 7, 0, 3, 6, 1, 4],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[7, 6, 5, 4, 3, 2, 1, 0],
        ],
        Case::Binary => &[&[3, 1]],
    };
    let tables: Vec<Vec<F>> = table_values
        .iter()
        .map(|table| table.iter().map(|&value| F::from_usize(value).unwrap()).collect())
        .collect();
    let p = tables[0].len();
    let plan = fbt_plan(params.base2k);
    let coeffs_meta = CoeffsMeta::from_delta_budget(INPUT_LOG_DELTA, params.base2k);

    let host_luts = match case {
        Case::General => HostLuts::Encoded(
            EncodedLut::general(
                host_module,
                &tables[0],
                params.base2k.into(),
                coeffs_meta,
                SplitStrategy::MinDepth,
            )
            .unwrap(),
        ),
        Case::Multi => HostLuts::Multi(
            tables
                .iter()
                .map(|table| {
                    EncodedLut::general(host_module, table, params.base2k.into(), coeffs_meta, SplitStrategy::MinDepth).unwrap()
                })
                .collect(),
        ),
        Case::Binary => HostLuts::Encoded(
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
        ),
    };
    let log_msg_ratio = match &host_luts {
        HostLuts::Encoded(lut) => lut.log_msg_ratio(),
        HostLuts::Multi(luts) => luts[0].log_msg_ratio(),
    };
    let log_modulus_in = INPUT_LOG_DELTA + log_msg_ratio;
    let lut_consumed = match &host_luts {
        HostLuts::Encoded(lut) => lut.consumed_bits(log_modulus_in, log_modulus_in),
        HostLuts::Multi(luts) => luts[0].consumed_bits(log_modulus_in, log_modulus_in),
    };
    let k_in = plan.input_k(log_modulus_in);
    let k_boot = plan
        .bootstrap_k(log_modulus_in + lut_consumed + 2 * INPUT_LOG_DELTA)
        .next_multiple_of(2 * params.base2k);
    let tp = CKKSTestParams {
        k: k_boot,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: INPUT_LOG_DELTA,
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
    let boot_tmp = module.ckks_bootstrap_tmp_bytes(
        &ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, k_boot - INPUT_LOG_DELTA),
        &ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, k_in - INPUT_LOG_DELTA),
        &ctx,
        &keys_layout,
    );
    if boot_tmp > initial_tmp {
        scratch = ScratchOwned::<BE>::alloc(boot_tmp);
    }

    let (sk_raw, sk) = gen_sk_with_raw(&tp, module, host_module, [0u8; 32]);
    let (mut xs, mut xe, mut xa) = (Source::new([7u8; 32]), Source::new([2u8; 32]), Source::new([1u8; 32]));
    let keys = ctx
        .generate_keys(
            module,
            host_module,
            &sk_raw,
            &keys_layout,
            &mut xs,
            &mut xe,
            &mut xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(module, &mut scratch.borrow());

    let mut source = Source::new([9u8; 32]);
    let sample = |source: &mut Source| ((source.next_f64(0.0, 1.0) * p as f64) as usize).min(p - 1);
    let (messages_re, messages_im): (Vec<usize>, Vec<usize>) =
        (0..params.n / 2).map(|_| (sample(&mut source), sample(&mut source))).unzip();
    let re: Vec<F> = messages_re.iter().map(|&value| F::from_usize(value).unwrap()).collect();
    let im: Vec<F> = messages_im.iter().map(|&value| F::from_usize(value).unwrap()).collect();
    let ct = ckks_encrypt_with_prec(
        &tp,
        module,
        host_module,
        &encoder,
        &sk,
        k_in,
        &re,
        &im,
        ckks_spec(params.n, params.base2k, INPUT_LOG_DELTA, k_in - INPUT_LOG_DELTA),
        &mut scratch.borrow(),
    );

    let outputs = match &host_luts {
        HostLuts::Encoded(lut) => {
            let lut = lut.map(|pt| upload_pt(module, pt));
            let mut output = module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into());
            module
                .ckks_functional_bootstrap(&mut output, &ct, &ctx, &lut, &keys, &mut scratch.borrow())
                .unwrap();
            vec![output]
        }
        HostLuts::Multi(luts) => {
            let luts: Vec<_> = luts.iter().map(|lut| lut.map(|pt| upload_pt(module, pt))).collect();
            let mut outputs: Vec<_> = luts
                .iter()
                .map(|_| module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into()))
                .collect();
            module
                .ckks_functional_bootstrap_multi(&mut outputs, &ct, &ctx, &luts, &keys, &mut scratch.borrow())
                .unwrap();
            outputs
        }
    };

    if matches!(case, Case::General) {
        let wrong_ratio_lut = EncodedLut::binary(
            host_module,
            F::from_usize(0).unwrap(),
            F::from_usize(1).unwrap(),
            EXP_DEGREE,
            K_INTERVAL,
            LOG_INTERVAL_REDUCTION,
            params.base2k.into(),
            coeffs_meta,
            SplitStrategy::MinDepth,
        )
        .unwrap()
        .map(|pt| upload_pt(module, pt));
        let mut output = module.ckks_ciphertext_alloc(params.base2k.into(), k_boot.into());
        let error = module
            .ckks_functional_bootstrap(&mut output, &ct, &ctx, &wrong_ratio_lut, &keys, &mut scratch.borrow())
            .unwrap_err();
        assert!(error.to_string().contains("log_msg_ratio"));
    }

    for (index, (output, table)) in outputs.iter().zip(&tables).enumerate() {
        let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(&tp, module, &encoder, output, &sk, &mut scratch.borrow());
        for (got, messages, part) in [(&got_re, &messages_re, "re"), (&got_im, &messages_im, "im")] {
            let want: Vec<F> = messages.iter().map(|&message| table[message]).collect();
            let stats = precision_stats(got, &want, output.log_delta());
            assert!(
                stats.avg_log2_prec >= 5.0,
                "functional bootstrapping [{index}] ({part}) averaged {:.1} bits",
                stats.avg_log2_prec
            );
        }
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
