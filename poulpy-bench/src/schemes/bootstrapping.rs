//! One-shot full-slot CKKS bootstrapping benchmark.

use std::{hint::black_box, time::Instant};

use poulpy_ckks::{
    CKKSCtBounds, CKKSMeta, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps},
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType,
        EncapsulationKeysLayout, SparseSecretEncapsulation,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, ckks_decrypt_decode, ckks_encrypt_with_prec, ckks_spec,
            gen_sk_with_raw, precision_stats, test_vector_1,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

const N: usize = 1 << 16;
const LOG_N: usize = 16;
const BASE2K: usize = 52;
const LOG_DELTA: usize = 38;
const LOG_MSG_RATIO: usize = 11;
const DSIZE: usize = 4;
const SECRET_WEIGHT: usize = 1024;
const EPHEMERAL_SECRET_WEIGHT: usize = 32;
const OVERFLOW_BOUND: usize = 16;
const RESTORED_LEVELS: usize = 16;
const MIN_AVG_PRECISION: f64 = 20.0;
const MAX_SECURE_MODULUS: usize = 1714;

fn meta(log_delta: usize, log_budget: usize) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(log_delta, log_budget)
}

fn plan() -> BootstrappingPlan {
    let slots_to_coeffs = DFTPlan::new(
        DFTType::Decode,
        vec![(3, 4), (4, 32), (4, 512), (4, 8192)],
        DFTOutputFormat::SplitRealAndImag,
        meta(31, 2),
    )
    .unwrap()
    .with_scaling(0.5)
    .unwrap();
    let coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        vec![(4, 8192), (4, 512), (4, 32), (3, 4)],
        DFTOutputFormat::SplitRealAndImag,
        meta(52, 2),
    )
    .unwrap();
    let eval_mod = EvalModPlan {
        eval_mod_type: EvalModType::CosHKEven,
        log_msg_ratio: LOG_MSG_RATIO,
        f_mod_degree: 30,
        f_mod_interval: OVERFLOW_BOUND,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: meta(42, 4),
        f_mod_log_delta: 58,
    };

    BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: EPHEMERAL_SECRET_WEIGHT,
            }),
            eval_round_plus: None,
        },
        coeffs_to_slots,
        eval_mod,
        slots_to_coeffs,
    )
    .unwrap()
}

pub fn run<BE, E>(name: &str, threads: usize, repeats: usize)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    E: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    assert!(threads > 0 && repeats > 0);
    let plan = plan();
    let log_modulus_in = LOG_DELTA + LOG_MSG_RATIO;
    let k_in = plan.input_k(log_modulus_in);
    let k_boot = plan.consumed_bits() + (RESTORED_LEVELS + 1) * LOG_DELTA;
    let gadget_width = DSIZE * BASE2K;
    let key_dnum = (k_boot + gadget_width) / gadget_width;
    let k_aux = gadget_width + LOG_N;
    let key_k = key_dnum * gadget_width + k_aux;
    assert!(k_boot <= MAX_SECURE_MODULUS && k_aux <= MAX_SECURE_MODULUS && key_k <= MAX_SECURE_MODULUS);

    let params = CKKSTestParams {
        n: N,
        base2k: BASE2K,
        k: k_boot,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: LOG_DELTA,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 8,
        hw: SECRET_WEIGHT,
        dsize: DSIZE,
        rank: 1,
    };
    let module = Module::<BE>::new(N as u64);
    let host_module = Module::<HostBytesBackend>::new(N as u64);
    let encoder = ReferenceEncoder::<E>::new::<f64>(N / 2).unwrap();

    let scratch_size = {
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), k_boot.into());
        ct.set_meta(params.prec().meta);
        module.ckks_all_ops_with_atk_tmp_bytes(
            &ct,
            &params.tsk_layout(),
            &params.atk_layout(),
            &ckks_spec(
                N,
                BASE2K,
                plan.eval_mod().coeffs_meta.log_delta(),
                plan.eval_mod().coeffs_meta.log_budget(),
            ),
        )
    };
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
    let context = BootstrappingContext::<BE, f64>::compile(&module, BASE2K.into(), &plan, &mut scratch.borrow()).unwrap();
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: params.atk_layout().layout,
        tensor_key: params.tsk_layout().layout,
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: params.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: params.ksk_layout(k_boot).layout,
        }),
    };
    let boot_scratch = module.ckks_bootstrap_tmp_bytes(
        &ckks_spec(N, BASE2K, LOG_DELTA, k_boot - LOG_DELTA),
        &ckks_spec(N, BASE2K, LOG_DELTA, k_in - LOG_DELTA),
        &context,
        &keys_layout,
    );
    if boot_scratch > scratch_size {
        scratch = ScratchOwned::<BE>::alloc(boot_scratch);
    }

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0; 32]);
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    let keys = context
        .generate_keys(
            &module,
            &sk_raw,
            &keys_layout,
            &mut source_xs,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());

    let (want_re, want_im) = test_vector_1::<f64>(N / 2);
    let input = ckks_encrypt_with_prec(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        k_in,
        &want_re,
        &want_im,
        ckks_spec(N, BASE2K, LOG_DELTA, k_in - LOG_DELTA),
        &mut scratch.borrow(),
    );
    let mut output = module.ckks_ciphertext_alloc(BASE2K.into(), k_boot.into());
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        output.set_k(k_boot.into());
        let start = Instant::now();
        module
            .ckks_bootstrap(
                black_box(&mut output),
                black_box(&input),
                black_box(&context),
                black_box(&keys),
                &mut scratch.borrow(),
            )
            .unwrap();
        samples.push(start.elapsed());
    }
    let (got_re, got_im) = ckks_decrypt_decode(&params, &module, &encoder, &output, &sk, &mut scratch.borrow());
    let re_precision = precision_stats(&got_re, &want_re, LOG_DELTA);
    let im_precision = precision_stats(&got_im, &want_im, LOG_DELTA);
    samples.sort_unstable();
    let min = samples[0].as_secs_f64();
    let median = samples[samples.len() / 2].as_secs_f64();
    let mean = samples.iter().map(|sample| sample.as_secs_f64()).sum::<f64>() / samples.len() as f64;
    println!(
        "RESULT backend={name} threads={} samples={repeats} min={min:.3}s median={median:.3}s mean={mean:.3}s \
         eval_mod={:?} degree={} interval={} split={:?} k_in={k_in} k_boot={k_boot} k_aux={k_aux} key_k={key_k} consumed={} restored_levels={RESTORED_LEVELS} \
         overflow=[-{OVERFLOW_BOUND},{OVERFLOW_BOUND}] \
         precision_re_avg={:.2}b precision_re_min={:.2}b precision_im_avg={:.2}b precision_im_min={:.2}b",
        threads,
        plan.eval_mod().eval_mod_type,
        plan.eval_mod().f_mod_degree,
        plan.eval_mod().f_mod_interval,
        plan.eval_mod().split_strategy,
        plan.consumed_bits(),
        re_precision.avg_log2_prec,
        re_precision.min_log2_prec,
        im_precision.avg_log2_prec,
        im_precision.min_log2_prec,
    );
    assert!(re_precision.avg_log2_prec >= MIN_AVG_PRECISION);
    assert!(im_precision.avg_log2_prec >= MIN_AVG_PRECISION);
}
