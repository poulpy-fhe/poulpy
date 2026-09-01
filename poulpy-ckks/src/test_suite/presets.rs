//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.

use anyhow::Result;
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
        PrecisionStats, TestContextBackend, TestContextHostModule, TestContextModule, ckks_spec, precision_stats, test_vector_1,
    },
};

/// Plaintext budget bits (above `log_delta`) used to measure the output precision.
pub const PRECISION_LOG_BUDGET: usize = 8;

/// The digit shape a backend runs a preset at: the preset's nominal shape for
/// exact (NTT) backends, and `base2k = 19` with 7-limb digits for approximate
/// FFT64 backends, whose products cannot carry the nominal radix.
pub fn preset_for_backend<BE: Backend>(preset: &BootstrappingPreset) -> Result<BootstrappingPreset> {
    if BE::DFT_IS_EXACT {
        Ok(preset.clone())
    } else {
        preset.with_base2k(19)?.with_dsizes(7, 7)
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
/// The advertised precision is pinned at the nominal shape with `f64` DFT
/// matrices, so the assertion applies to exact (NTT) backends only; approximate
/// FFT64 backends run at a reduced radix and only report their measurement.
/// Full logN16 bootstraps are slow, so backends register this as an ignored test.
pub fn bootstrapping_presets_meet_precision<BE>()
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
    for preset in all().unwrap() {
        let preset = preset_for_backend::<BE>(&preset).unwrap();
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
        if BE::DFT_IS_EXACT {
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
}
