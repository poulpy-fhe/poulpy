//! Full-slot CKKS bootstrapping benchmark.

use std::hint::black_box;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps,
        CKKSEncryptOps,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques,
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout,
        SparseSecretEncapsulation,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, ckks_spec, gen_sk_with_raw, precision_stats,
            test_vector_1,
        },
    },
};
use poulpy_core::{
    EncryptionLayout,
    layouts::{
        GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::params::{CkksBootstrappingBenchParams, default_bench_params_ckks_bootstrapping};

const N: usize = 1 << 16;
const LOG_N: usize = 16;
const LOG_DELTA: usize = 38;
const LOG_MSG_RATIO: usize = 11;
const SECRET_WEIGHT: usize = 1024;
const EPHEMERAL_SECRET_WEIGHT: usize = 32;
const OVERFLOW_BOUND: usize = 16;
const RESTORED_LEVELS: usize = 16;
const MAX_SECURE_MODULUS: usize = 1714;
const MAX_DENSE_TO_SPARSE_MODULUS: usize = 349;

fn meta(log_delta: usize, log_budget: usize) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(log_delta, log_budget)
}

fn plan() -> BootstrappingPlan {
    let slots_to_coeffs = DFTPlan::new(
        DFTType::Decode,
        vec![(3, 4), (4, 32), (4, 512), (4, 8192)],
        DFTOutputFormat::SplitRealAndImag,
        meta(28, 2),
    )
    .unwrap()
    .with_scaling(0.5)
    .unwrap();
    let coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        vec![(4, 8192), (4, 512), (4, 32), (3, 4)],
        DFTOutputFormat::SplitRealAndImag,
        meta(44, 3),
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

fn runner_ckks_bootstrapping<BE>(group: &mut BenchmarkGroup<'_, WallTime>, config: CkksBootstrappingBenchParams)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let CkksBootstrappingBenchParams {
        base2k,
        dsize,
        dense_to_sparse_dsize,
    } = config;
    let mut state = None;
    let mut precision = None;
    group.bench_with_input(BenchmarkId::from_parameter(config), &config, |bencher, _| {
        let (module, context, keys, scratch, input, output, sk, want_re, want_im, k_boot, prec_log_budget) = state
            .get_or_insert_with(|| {
                assert!(base2k > 0 && dsize > 0 && dense_to_sparse_dsize > 0);
                let plan = plan();
                let log_modulus_in = LOG_DELTA + LOG_MSG_RATIO;
                let k_in = plan.input_k(log_modulus_in);
                let k_boot = plan.consumed_bits() + (RESTORED_LEVELS + 1) * LOG_DELTA;
                let gadget_width = dsize * base2k;
                let key_dnum = (k_boot + gadget_width) / gadget_width;
                let k_aux = gadget_width + LOG_N;
                let key_k = key_dnum * gadget_width + k_aux;
                assert!(k_boot <= MAX_SECURE_MODULUS && k_aux <= MAX_SECURE_MODULUS && key_k <= MAX_SECURE_MODULUS);

                let params = CKKSTestParams {
                    n: N,
                    base2k,
                    k: k_boot,
                    prec_meta: CKKSMeta {
                        log_sparsity: 0,
                        log_delta: LOG_DELTA,
                        slots: SlotsKind::Complex,
                    },
                    prec_log_budget: 8,
                    hw: SECRET_WEIGHT,
                    dsize,
                    rank: 1,
                };
                let module = Module::<BE>::new(N as u64);
                let host_module = Module::<HostBytesBackend>::new(N as u64);

                let scratch_size = {
                    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
                    ct.set_meta(params.prec().meta);
                    module.ckks_all_ops_with_atk_tmp_bytes(
                        &ct,
                        &params.tsk_layout(),
                        &params.atk_layout(),
                        &ckks_spec(
                            N,
                            base2k,
                            plan.eval_mod().coeffs_meta.log_delta(),
                            plan.eval_mod().coeffs_meta.log_budget(),
                        ),
                    )
                };
                let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
                let context =
                    BootstrappingContext::<BE, f64>::compile(&module, base2k.into(), &plan, &mut scratch.borrow()).unwrap();
                let dense_to_sparse = CKKSTestParams {
                    dsize: dense_to_sparse_dsize,
                    ..params
                }
                .ksk_layout(log_modulus_in)
                .layout;
                let dense_to_sparse_modulus = dense_to_sparse.k().as_usize();
                assert!(
                    dense_to_sparse_modulus <= MAX_DENSE_TO_SPARSE_MODULUS,
                    "dense-to-sparse key modulus {dense_to_sparse_modulus} exceeds {MAX_DENSE_TO_SPARSE_MODULUS}"
                );
                let keys_layout = BootstrappingKeysLayout {
                    automorphism_key: params.atk_layout().layout,
                    tensor_key: params.tsk_layout().layout,
                    encapsulation: Some(EncapsulationKeysLayout {
                        dense_to_sparse,
                        sparse_to_dense: params.ksk_layout(k_boot).layout,
                    }),
                };
                let boot_scratch = module.ckks_bootstrap_tmp_bytes(
                    &ckks_spec(N, base2k, LOG_DELTA, k_boot - LOG_DELTA),
                    &ckks_spec(N, base2k, LOG_DELTA, k_in - LOG_DELTA),
                    &context,
                    &keys_layout,
                );
                if boot_scratch > scratch_size {
                    scratch = ScratchOwned::<BE>::alloc(boot_scratch);
                }

                let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0; 32]);
                let (mut source_xs, mut source_xa, mut source_xe) =
                    (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
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
                let input_spec = ckks_spec(N, base2k, LOG_DELTA, k_in - LOG_DELTA);
                let mut input_pt = module.ckks_pt_vec_alloc(base2k.into(), input_spec.k());
                input_pt.set_meta(input_spec.meta());
                module
                    .ckks_encode_reim_into(&mut input_pt, &want_re, &want_im, &mut scratch.borrow())
                    .unwrap();
                let mut input_layout = params.glwe_layout().layout;
                input_layout.k = k_in.into();
                let input_enc_infos = EncryptionLayout::new_from_default_sigma(input_layout).unwrap();
                let mut input = module.ckks_ciphertext_alloc(base2k.into(), k_in.into());
                let (mut input_xa, mut input_xe) = (Source::new([3; 32]), Source::new([4; 32]));
                module
                    .ckks_encrypt_sk(
                        &mut input,
                        &input_pt,
                        &sk,
                        &input_enc_infos,
                        &mut input_xe,
                        &mut input_xa,
                        &mut scratch.borrow(),
                    )
                    .unwrap();
                let mut output = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
                module
                    .ckks_bootstrap(&mut output, &input, &context, &keys, &mut scratch.borrow())
                    .unwrap();
                (
                    module,
                    context,
                    keys,
                    scratch,
                    input,
                    output,
                    sk,
                    want_re,
                    want_im,
                    k_boot,
                    params.prec().log_budget(),
                )
            });
        bencher.iter(|| {
            output.set_k((*k_boot).into());
            module
                .ckks_bootstrap(
                    black_box(&mut *output),
                    black_box(&*input),
                    black_box(&*context),
                    black_box(&*keys),
                    &mut scratch.borrow(),
                )
                .unwrap();
        });
        let log_budget = output
            .log_budget()
            .min(*prec_log_budget)
            .min(127usize.saturating_sub(output.log_delta()));
        let mut output_pt = module.ckks_pt_vec_alloc(output.base2k(), (output.log_delta() + log_budget).into());
        output_pt.set_meta(CKKSMeta {
            log_sparsity: 0,
            log_delta: output.log_delta(),
            slots: SlotsKind::Complex,
        });
        module
            .ckks_decrypt(&mut output_pt, output, sk, &mut scratch.borrow())
            .unwrap();
        let (mut got_re, mut got_im) = (vec![0.0; N / 2], vec![0.0; N / 2]);
        module
            .ckks_decode_reim_into(&output_pt, &mut got_re, &mut got_im, &mut scratch.borrow())
            .unwrap();
        precision = Some((
            precision_stats(&got_re, want_re, LOG_DELTA),
            precision_stats(&got_im, want_im, LOG_DELTA),
        ));
    });
    if let Some((re_precision, im_precision)) = precision {
        let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
        println!(
            "PRECISION backend={backend} base2k={base2k} dsize={dsize} re_avg={:.2}b re_min={:.2}b im_avg={:.2}b im_min={:.2}b",
            re_precision.avg_log2_prec, re_precision.min_log2_prec, im_precision.avg_log2_prec, im_precision.min_log2_prec,
        );
    }
}

pub fn bench_ckks_bootstrapping<BE>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for params in default_bench_params_ckks_bootstrapping::<BE>() {
        runner_ckks_bootstrapping::<BE>(&mut group, params);
    }
    group.finish();
}
