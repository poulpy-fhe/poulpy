//! Full-slot CKKS bootstrapping benchmark.
//!
//! The benchmark sweep includes both logN16 presets. Criterion's name filter
//! can select `c2s_16_levels` or `s2c_16_levels` individually.

use std::hint::black_box;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps,
        CKKSEncryptOps,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned,
        EncapsulationKeysLayout,
    },
    presets::bootstrapping::{BootstrappingPreset, log_n16},
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

use super::params::{CkksBootstrappingBenchParams, CkksBootstrappingPreset, default_bench_params_ckks_bootstrapping};

fn preset(kind: CkksBootstrappingPreset) -> BootstrappingPreset {
    match kind {
        CkksBootstrappingPreset::C2S16Levels => log_n16::c2s_16_levels(),
        CkksBootstrappingPreset::S2C16Levels => log_n16::s2c_16_levels(),
    }
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
        preset: preset_kind,
        base2k,
        dsize,
        dense_to_sparse_dsize,
    } = config;
    let preset = preset(preset_kind);
    let plan = preset.plan();
    let n = preset.n();
    let log_delta = preset.log_delta();
    let log_modulus = preset.log_modulus();
    let k_in = preset.input_k();
    let k_boot = preset.bootstrap_k();
    let mut state = None;
    let mut precision = None;
    group.bench_with_input(BenchmarkId::from_parameter(config), &config, |bencher, _| {
        let (module, context, keys, scratch, input, output, sk, want_re, want_im, k_boot, prec_log_budget) = state
            .get_or_insert_with(|| {
                assert!(base2k > 0 && dsize > 0 && dense_to_sparse_dsize > 0);

                let params = CKKSTestParams {
                    n,
                    base2k,
                    k: k_boot,
                    prec_meta: CKKSMeta {
                        log_sparsity: 0,
                        log_delta,
                        slots: SlotsKind::Complex,
                    },
                    prec_log_budget: 8,
                    hw: preset.dense_secret_hamming_weight(),
                    dsize,
                    rank: 1,
                };
                let module = Module::<BE>::new(n as u64);
                let host_module = Module::<HostBytesBackend>::new(n as u64);

                let scratch_size = {
                    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
                    ct.set_meta(params.prec().meta);
                    module.ckks_all_ops_with_atk_tmp_bytes(
                        &ct,
                        &params.tsk_layout(),
                        &params.atk_layout(),
                        &ckks_spec(
                            n,
                            base2k,
                            plan.eval_mod().coeffs_meta.log_delta(),
                            plan.eval_mod().coeffs_meta.log_budget(),
                        ),
                    )
                };
                let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
                let context =
                    BootstrappingContext::<BE, f64>::compile(&module, base2k.into(), plan, &mut scratch.borrow()).unwrap();
                let dense_to_sparse = CKKSTestParams {
                    dsize: dense_to_sparse_dsize,
                    ..params
                }
                .ksk_layout(log_modulus)
                .layout;
                let keys_layout = if base2k == preset.base2k()
                    && dsize == preset.keys_layout().automorphism_key.dsize.as_usize()
                    && dense_to_sparse_dsize
                        == preset
                            .keys_layout()
                            .encapsulation
                            .as_ref()
                            .unwrap()
                            .dense_to_sparse
                            .dsize
                            .as_usize()
                {
                    *preset.keys_layout()
                } else {
                    BootstrappingKeysLayout {
                        automorphism_key: params.atk_layout().layout,
                        tensor_key: params.tsk_layout().layout,
                        encapsulation: Some(EncapsulationKeysLayout {
                            dense_to_sparse,
                            sparse_to_dense: params.ksk_layout(k_boot).layout,
                        }),
                    }
                };
                assert!(keys_layout.automorphism_key.k().as_usize() <= preset.max_dense_modulus());
                assert!(keys_layout.tensor_key.k().as_usize() <= preset.max_dense_modulus());
                let encapsulation = keys_layout.encapsulation.as_ref().unwrap();
                assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.max_sparse_modulus());
                assert!(encapsulation.sparse_to_dense.k().as_usize() <= preset.max_dense_modulus());
                let boot_scratch = module.ckks_bootstrap_tmp_bytes(
                    &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
                    &ckks_spec(n, base2k, log_delta, k_in - log_delta),
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

                let (want_re, want_im) = test_vector_1::<f64>(n / 2);
                let input_spec = ckks_spec(n, base2k, log_delta, k_in - log_delta);
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
                assert_eq!(output.k().as_usize(), preset.output_k());
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
        let (mut got_re, mut got_im) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        module
            .ckks_decode_reim_into(&output_pt, &mut got_re, &mut got_im, &mut scratch.borrow())
            .unwrap();
        precision = Some((
            precision_stats(&got_re, want_re, log_delta),
            precision_stats(&got_im, want_im, log_delta),
        ));
    });
    if let Some((re_precision, im_precision)) = precision {
        let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
        println!(
            "PRECISION backend={backend} preset={preset_kind} base2k={base2k} dsize={dsize} re_avg={:.2}b re_min={:.2}b re_worst_idx={} re_worst_err={:.3e} im_avg={:.2}b im_min={:.2}b im_worst_idx={} im_worst_err={:.3e}",
            re_precision.avg_log2_prec,
            re_precision.min_log2_prec,
            re_precision.worst_idx,
            re_precision.worst_err,
            im_precision.avg_log2_prec,
            im_precision.min_log2_prec,
            im_precision.worst_idx,
            im_precision.worst_err,
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

#[cfg(test)]
mod tests {
    use poulpy_ckks::layouts::BootstrappingPipeline;

    use super::*;

    #[test]
    fn preset_selector_covers_both_pipelines() {
        let c2s = preset(CkksBootstrappingPreset::C2S16Levels);
        let s2c = preset(CkksBootstrappingPreset::S2C16Levels);

        assert_eq!(c2s.plan().pipeline(), BootstrappingPipeline::C2SFirst);
        assert_eq!(s2c.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(c2s.restored_levels(), 16);
        assert_eq!(s2c.restored_levels(), 16);
    }
}
