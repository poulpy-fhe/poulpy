//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.

use poulpy_core::{
    EncryptionLayout,
    layouts::{
        GGLWEInfos, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPrepared, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps,
        CKKSEncryptOps,
    },
    layouts::{BootstrappingContext, BootstrappingKeysPrepared, CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::helpers::{
        PrecisionStats, TestContextBackend, TestContextHostModule, TestContextModule, assert_canonical_at_k, ckks_spec,
        precision_stats, test_vector_1,
    },
};

/// Plaintext budget bits (above `log_delta`) used to measure the output precision.
pub const PRECISION_LOG_BUDGET: usize = 8;

/// Caps the nominal preset at the caller's explicit test or benchmark radix.
///
/// This fixture choice carries no failure-probability guarantee. A smaller
/// radix uses up to 7 high-modulus limbs, reduced to fit the modulus bounds,
/// and 1 dense-to-sparse limb; the circuit and bit widths stay unchanged.
pub fn preset_with_max_base2k(preset: &BootstrappingPreset, fixture_base2k: usize) -> anyhow::Result<BootstrappingPreset> {
    if preset.base2k() <= fixture_base2k {
        Ok(preset.clone())
    } else {
        let preset = preset.with_base2k(fixture_base2k)?;
        for dsize in (2..=7).rev() {
            if let Ok(adapted) = preset.with_dsizes(dsize, 1) {
                return Ok(adapted);
            }
        }
        preset.with_dsizes(1, 1)
    }
}

/// A preset set up end to end and ready to bootstrap repeatedly.
pub struct BootstrappingPresetRun<BE: Backend> {
    preset: BootstrappingPreset,
    module: Module<BE>,
    context: BootstrappingContext<BE, f64>,
    keys: BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    scratch: ScratchOwned<BE>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    want_re: Vec<f64>,
    want_im: Vec<f64>,
}

impl<BE> BootstrappingPresetRun<BE>
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
    /// Compiles the context, generates the keys, encrypts the reference vector
    /// at the preset's input layout, and runs one bootstrap to check the output
    /// width. Every layout comes from `preset`.
    pub fn setup(preset: BootstrappingPreset) -> Self {
        let plan = preset.plan();
        let n = preset.n();
        let base2k = preset.base2k();
        let input_layout = preset.input_layout();
        let bootstrap_layout = preset.bootstrap_layout();
        let keys_layout = *preset.keys_layout();
        let module = Module::<BE>::new(n as u64);

        let scratch_size = {
            let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);
            ct.set_meta(bootstrap_layout.meta);
            module.ckks_all_ops_with_atk_tmp_bytes(
                &ct,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                &ckks_spec(
                    n,
                    base2k,
                    plan.eval_mod().coeffs_meta.log_delta(),
                    plan.eval_mod().coeffs_meta.log_budget(),
                ),
            )
        };
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
        let context = BootstrappingContext::<BE, f64>::compile(&module, base2k.into(), plan, &mut scratch.borrow()).unwrap();
        let boot_scratch = module.ckks_bootstrap_tmp_bytes(&bootstrap_layout, &input_layout, &context, &keys_layout);
        if boot_scratch > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(boot_scratch);
        }

        // Dense application secret at the preset's Hamming weight.
        let mut source_sk = Source::new([0; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut source_sk);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);

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

        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut input_pt = module.ckks_pt_vec_alloc(base2k.into(), input_layout.k());
        input_pt.set_meta(input_layout.meta());
        module
            .ckks_encode_reim_into(&mut input_pt, &want_re, &want_im, &mut scratch.borrow())
            .unwrap();
        let input_enc_infos = EncryptionLayout::new_from_default_sigma(input_layout.glwe_layout).unwrap();
        let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
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
        let output = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);

        let mut run = Self {
            preset,
            module,
            context,
            keys,
            scratch,
            input,
            output,
            sk,
            want_re,
            want_im,
        };
        run.bootstrap();
        assert_eq!(run.output.k().as_usize(), run.preset.output_k());
        assert_eq!(run.output.meta(), run.preset.output_layout().meta());
        assert_eq!(run.output.log_delta(), run.input.log_delta());
        assert_eq!(
            run.output.k().as_usize() - run.input.k().as_usize(),
            run.preset.output_k() - run.preset.input_k()
        );
        run
    }

    /// The preset being exercised.
    pub fn preset(&self) -> &BootstrappingPreset {
        &self.preset
    }

    /// Bootstraps the reference input into the output ciphertext.
    pub fn bootstrap(&mut self) {
        self.output.set_k(self.preset.bootstrap_k().into());
        self.module
            .ckks_bootstrap(
                &mut self.output,
                &self.input,
                &self.context,
                &self.keys,
                &mut self.scratch.borrow(),
            )
            .unwrap();
    }

    /// Decrypts the last bootstrap output and measures its precision against
    /// the reference vector, as `(real, imaginary)` statistics.
    pub fn precision(&mut self) -> (PrecisionStats, PrecisionStats) {
        let output = &self.output;
        assert_canonical_at_k::<BE>("bootstrap preset", output);
        let log_budget = output
            .log_budget()
            .min(PRECISION_LOG_BUDGET)
            .min(127usize.saturating_sub(output.log_delta()));
        let mut output_pt = self
            .module
            .ckks_pt_vec_alloc(output.base2k(), (output.log_delta() + log_budget).into());
        output_pt.set_meta(CKKSMeta {
            log_sparsity: 0,
            log_delta: output.log_delta(),
            slots: SlotsKind::Complex,
        });
        self.module
            .ckks_decrypt(&mut output_pt, output, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let n = self.preset.n();
        let (mut got_re, mut got_im) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        self.module
            .ckks_decode_reim_into(&output_pt, &mut got_re, &mut got_im, &mut self.scratch.borrow())
            .unwrap();
        let log_delta = self.preset.log_delta();
        (
            precision_stats(&got_re, &self.want_re, log_delta),
            precision_stats(&got_im, &self.want_im, log_delta),
        )
    }
}

/// Runs every preset once on `BE` and checks the measured output precision
/// against the precision the preset advertises.
///
/// The caller supplies the fixture radix limit; this precision check does
/// not establish a failure-probability bound. Every backend uses `f64` DFT
/// matrices and must reach the advertised precision. Full-size bootstraps are
/// slow, so backends register this as an ignored test.
pub fn bootstrapping_presets_meet_precision<BE>(fixture_base2k: usize)
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
    let backend = std::any::type_name::<BE>();
    for preset in all().unwrap() {
        let preset = preset_with_max_base2k(&preset, fixture_base2k).unwrap();
        let mut run = BootstrappingPresetRun::<BE>::setup(preset);
        let (re, im) = run.precision();
        let preset = run.preset();
        println!(
            "PRECISION backend={backend} preset={} base2k={} re_min={:.2}b re_avg={:.2}b im_min={:.2}b im_avg={:.2}b (advertised {}b)",
            preset.name(),
            preset.base2k(),
            re.min_log2_prec,
            re.avg_log2_prec,
            im.min_log2_prec,
            im.avg_log2_prec,
            preset.log2_precision(),
        );
        let advertised = preset.log2_precision() as f64;
        assert!(
            re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
            "preset {} advertises {advertised} bits but measured re_min={:.2} im_min={:.2}",
            preset.name(),
            re.min_log2_prec,
            im.min_log2_prec
        );
    }
}

/// Checks the advertised precision on both CI bootstrapping paths.
/// Backends register these full-size checks as ignored tests.
pub fn ci_bootstrapping_preset_meets_precision<BE, STD>(
    ci: Module<BE>,
    standard: Module<STD>,
    preset: crate::presets::bootstrapping::CIBootstrappingPreset,
    fixture_base2k: usize,
) where
    BE: TestContextBackend,
    STD: TestContextBackend,
    Module<STD>: TestContextModule<STD> + CKKSEncodingOps<STD, f64> + CKKSBootstrappingOps<STD> + CKKSDFTMatrixOps<STD, f64>,
    for<'a> STD::BufRef<'a>: HostDataRef,
    for<'a> STD::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let preset = if preset.base2k() > fixture_base2k {
        preset.with_base2k(fixture_base2k).unwrap().with_dsizes(7, 1, 1).unwrap()
    } else {
        preset
    };
    let params = crate::test_suite::CKKSTestParams {
        ring_kind: crate::CKKSRingKind::ConjugateInvariant,
        n: preset.n(),
        base2k: preset.base2k(),
        k: preset.bootstrap_k(),
        prec_meta: preset.input_layout().meta,
        prec_log_budget: preset.input_k() - preset.log_delta(),
        hw: preset.dense_secret_hamming_weight(),
        dsize: preset.keys_layout().bootstrap_keys.automorphism_key.dsize.as_usize(),
        rank: 1,
    };
    let mut run = CIBootstrappingRun::setup(
        ci,
        standard,
        preset.plan(),
        params,
        *preset.keys_layout(),
        preset.input_k(),
        preset.output_k(),
    );
    for pair in [false, true] {
        run.bootstrap(pair);
        for (index, precision) in run.precision(pair).iter().enumerate() {
            println!(
                "PRECISION backend={} preset={} base2k={} pair={pair} output={index} min={:.2}b avg={:.2}b",
                std::any::type_name::<BE>(),
                preset.name(),
                preset.base2k(),
                precision.min_log2_prec,
                precision.avg_log2_prec
            );
            assert!(
                precision.min_log2_prec >= preset.log2_precision() as f64,
                "{} precision {:.2} below {} bits",
                preset.name(),
                precision.min_log2_prec,
                preset.log2_precision()
            );
        }
    }
}

/// End-to-end fixture for CI bootstrapping conformance tests.
pub(crate) struct CIBootstrappingRun<BE: Backend, STD: Backend> {
    pub(crate) ci: Module<BE>,
    pub(crate) standard: Module<STD>,
    pub(crate) context: crate::layouts::CIBootstrappingContext<STD, f64>,
    pub(crate) keys: crate::layouts::CIBootstrappingKeysPrepared<STD::OwnedBuf, STD>,
    pub(crate) scratch: ScratchOwned<BE>,
    pub(crate) std_scratch: ScratchOwned<STD>,
    pub(crate) inputs: [CKKSCiphertextOwned<BE>; 2],
    pub(crate) outputs: [CKKSCiphertextOwned<BE>; 2],
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: [Vec<f64>; 2],
    bootstrap_k: usize,
    output_k: usize,
}

impl<BE, STD> CIBootstrappingRun<BE, STD>
where
    BE: TestContextBackend,
    STD: TestContextBackend,
    Module<STD>: TestContextModule<STD> + CKKSEncodingOps<STD, f64> + CKKSBootstrappingOps<STD> + CKKSDFTMatrixOps<STD, f64>,
    for<'a> STD::BufRef<'a>: HostDataRef,
    for<'a> STD::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    /// Compiles a standard context, generates independent CI and standard secrets,
    /// and encrypts two distinct real vectors for single and pair evaluation.
    pub fn setup(
        ci: Module<BE>,
        standard: Module<STD>,
        plan: &crate::layouts::BootstrappingPlan,
        params: crate::test_suite::CKKSTestParams,
        keys_layout: crate::layouts::CIBootstrappingKeysLayout,
        input_k: usize,
        output_k: usize,
    ) -> Self {
        use super::helpers::{alloc_scratch, gen_sk_with_raw};
        use poulpy_core::layouts::GLWEInfos;
        let standard_host = Module::<HostBytesBackend>::new(standard.n() as u64);
        let ci_host = Module::<HostBytesBackend>::new(ci.n() as u64);
        let standard_params = crate::test_suite::CKKSTestParams {
            ring_kind: crate::CKKSRingKind::Standard,
            n: standard.n(),
            ..params
        };
        let mut std_scratch = alloc_scratch(&standard_params, &standard);
        let mut scratch = alloc_scratch(&params, &ci);
        let context =
            crate::layouts::CIBootstrappingContext::compile(&standard, params.base2k.into(), plan, &mut std_scratch.borrow())
                .unwrap();
        let (standard_sk, _) = gen_sk_with_raw(&standard_params, &standard, &standard_host, [11; 32]);
        let (ci_sk, sk) = gen_sk_with_raw(&params, &ci, &ci_host, [12; 32]);
        let mut source_xs = Source::new([13; 32]);
        let mut source_xe = Source::new([14; 32]);
        let mut source_xa = Source::new([15; 32]);
        let keys = context
            .generate_keys(
                &standard,
                &ci_sk,
                &standard_sk,
                &keys_layout,
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut std_scratch.borrow(),
            )
            .unwrap()
            .prepare(&standard, &mut std_scratch.borrow())
            .unwrap();
        let slots = ci.n() >> params.prec_meta.log_sparsity;
        let (want0, want1) = test_vector_1::<f64>(slots);
        let want = [want0, want1];
        let inputs = std::array::from_fn(|index| {
            let mut pt = ci.ckks_pt_vec_alloc_compact(slots, params.base2k.into(), input_k.into());
            pt.set_meta(params.prec_meta);
            ci.ckks_encode_reim_into(&mut pt, &want[index], &vec![0.0; slots], &mut scratch.borrow())
                .unwrap();
            let mut ct = ci.ckks_ciphertext_alloc(params.base2k.into(), input_k.into());
            let enc = EncryptionLayout::new_from_default_sigma(ct.glwe_layout()).unwrap();
            ci.ckks_encrypt_sk(&mut ct, &pt, &sk, &enc, &mut source_xe, &mut source_xa, &mut scratch.borrow())
                .unwrap();
            ct
        });
        let outputs = std::array::from_fn(|_| ci.ckks_ciphertext_alloc(params.base2k.into(), params.k.into()));
        let bytes = crate::layouts::CIRingBridge::new(&ci, &standard)
            .unwrap()
            .bootstrap_tmp_bytes(&outputs[0], &inputs[0], &context, &keys_layout);
        std_scratch = ScratchOwned::<STD>::alloc(bytes);
        Self {
            ci,
            standard,
            context,
            keys,
            scratch,
            std_scratch,
            inputs,
            outputs,
            sk,
            want,
            bootstrap_k: params.k,
            output_k,
        }
    }

    /// Runs the single-ciphertext path, or the explicit pair path when requested.
    pub fn bootstrap(&mut self, pair: bool) {
        for ct in &mut self.outputs {
            ct.set_k(self.bootstrap_k.into());
        }
        let [left, right] = &mut self.outputs;
        if pair {
            crate::layouts::CIRingBridge::new(&self.ci, &self.standard)
                .unwrap()
                .bootstrap_pair(
                    left,
                    right,
                    &self.inputs[0],
                    &self.inputs[1],
                    &self.context,
                    &self.keys,
                    &mut self.std_scratch.borrow(),
                )
                .unwrap();
        } else {
            crate::layouts::CIRingBridge::new(&self.ci, &self.standard)
                .unwrap()
                .bootstrap(
                    left,
                    &self.inputs[0],
                    &self.context,
                    &self.keys,
                    &mut self.std_scratch.borrow(),
                )
                .unwrap();
        }
    }

    /// Decrypts and measures every output produced by the selected path.
    pub fn precision(&mut self, pair: bool) -> Vec<PrecisionStats> {
        self.outputs
            .iter()
            .zip(&self.want)
            .take(if pair { 2 } else { 1 })
            .map(|(ct, want)| {
                assert_eq!(ct.k().as_usize(), self.output_k);
                assert_eq!(ct.meta(), self.inputs[0].meta());
                assert_eq!(ct.ring_kind(), crate::CKKSRingKind::ConjugateInvariant);
                assert_canonical_at_k::<BE>("CI bootstrap", ct);
                let mut pt = self.ci.ckks_pt_vec_alloc(ct.base2k(), ct.k());
                pt.set_meta(ct.meta());
                self.ci
                    .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())
                    .unwrap();
                use poulpy_hal::layouts::ZnxView;
                let view = GLWEToBackendRef::<BE>::to_backend_ref(&pt);
                let high_limbs = ct.log_budget().saturating_sub(PRECISION_LOG_BUDGET) / ct.base2k().as_usize();
                for limb in 0..high_limbs {
                    assert!(
                        view.data().at(0, limb).iter().all(|&x| x == 0),
                        "CI bootstrap has a high-modulus alias"
                    );
                }
                let mut pt = self
                    .ci
                    .ckks_pt_vec_alloc(ct.base2k(), (ct.log_delta() + PRECISION_LOG_BUDGET).into());
                pt.set_meta(ct.meta());
                self.ci
                    .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())
                    .unwrap();
                let (mut re, mut im) = (vec![0.0; want.len()], vec![0.0; want.len()]);
                self.ci
                    .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.scratch.borrow())
                    .unwrap();
                assert!(im.iter().all(|&v| v == 0.0));
                assert!(re.iter().all(|v| v.is_finite()));
                precision_stats(&re, want, ct.log_delta())
            })
            .collect()
    }
}
