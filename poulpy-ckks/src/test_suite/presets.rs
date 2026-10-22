//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.
//! [`BootstrappingCIPresetRun`] does the same for the presets with EvalMod on the
//! conjugate-invariant ring.

use crate::CKKSCompositionError;
use crate::CKKSModuleInfos;
use crate::Scale;
use poulpy_core::layouts::{
    GGLWEInfos, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPrepared, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, SetK, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ReaderFrom, ScratchOwned, Standard, WriterTo},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDecryptOps, CKKSEncodingHostOps,
        CKKSEncodingOps, CKKSEncryptOps,
    },
    layouts::{
        BootstrappingCIContext, BootstrappingCIKeysPrepared, BootstrappingContext, BootstrappingKeysLayout,
        BootstrappingKeysPrepared, BootstrappingPlan, CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned,
    },
    presets::bootstrapping::{BootstrappingCIPreset, BootstrappingPreset, all, all_ci},
    test_suite::helpers::{
        PrecisionStats, TestContextBackend, TestContextModule, assert_canonical_at_k, ckks_spec, precision_stats, test_vector_1,
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
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
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
        Self::setup_with_roundtrip(preset, false)
    }

    fn setup_with_roundtrip(preset: BootstrappingPreset, roundtrip_keys: bool) -> Self {
        let plan = preset.plan();
        let n = preset.n();
        let base2k = preset.base2k();
        let input_layout = preset.input_layout();
        let bootstrap_layout = preset.bootstrap_layout();
        let keys_layout = preset.keys_layout().clone();
        let module = Module::<BE>::new(n as u64);

        let scratch_size = bootstrap_setup_tmp_bytes(&module, &bootstrap_layout, plan, &keys_layout);
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
        let unprepared =
            BootstrappingContext::<BE, f64>::compile_unprepared(&module, base2k.into(), plan, &mut scratch.borrow()).unwrap();
        let prepare_scratch = unprepared.prepare_tmp_bytes(&module);
        if prepare_scratch > BE::len_bytes(&scratch.data) {
            scratch = ScratchOwned::<BE>::alloc(prepare_scratch);
        }
        let context = unprepared.prepare(&module, &mut scratch.borrow());
        let boot_scratch = module.ckks_bootstrap_tmp_bytes(&bootstrap_layout, &input_layout, &context, &keys_layout);
        if boot_scratch > BE::len_bytes(&scratch.data) {
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
            .unwrap();
        if roundtrip_keys {
            fn roundtrip<T: ReaderFrom + WriterTo + PartialEq>(key: &T, mut restored: T) {
                let mut bytes = Vec::new();
                key.write_to(&mut bytes).unwrap();
                restored.read_from(&mut bytes.as_slice()).unwrap();
                assert!(restored == *key);
            }
            for key in keys.rotation_keys.values() {
                roundtrip(key, module.glwe_automorphism_key_alloc_from_infos(key));
            }
            roundtrip(&keys.tensor_key, module.glwe_tensor_key_alloc_from_infos(&keys.tensor_key));
            let (dense_to_sparse, sparse_to_dense) = keys.encapsulation_keys.as_ref().expect("preset needs encapsulation keys");
            roundtrip(dense_to_sparse, module.glwe_switching_key_alloc_from_infos(dense_to_sparse));
            if let Some(sparse_to_dense) = sparse_to_dense {
                roundtrip(sparse_to_dense, module.glwe_switching_key_alloc_from_infos(sparse_to_dense));
            }
            for key in keys.first_factor_keys.values() {
                roundtrip(key, module.glwe_switching_key_alloc_from_infos(key));
            }
        }
        let keys = keys.prepare(&module, &mut scratch.borrow());

        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut input_pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), base2k.into(), input_layout.k());
        input_pt.set_meta(input_layout.meta());
        module
            .ckks_encode_reim_into(&mut input_pt, &want_re, &want_im, &mut scratch.borrow())
            .unwrap();

        let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        let (mut input_xa, mut input_xe) = (Source::new([3; 32]), Source::new([4; 32]));
        module
            .ckks_encrypt_sk(
                &mut input,
                &input_pt,
                &sk,
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
        let mut output_pt = self.module.ckks_pt_vec_alloc(
            self.module.ckks_max_slots(),
            output.base2k(),
            (output.log_delta() + log_budget).into(),
        );
        output_pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(output.log_delta()),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
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
/// against the precision the preset advertises. The first preset, and the
/// first with the first CoeffsToSlots factor on the raise, also round-trip
/// every generated key, both encapsulation directions and the first-factor
/// keys included.
///
/// The caller supplies the fixture radix limit; this precision check does
/// not establish a failure-probability bound. Every backend uses `f64` DFT
/// matrices and must reach the advertised precision. Full-size bootstraps are
/// slow, so backends register this as an ignored test.
pub fn bootstrapping_presets_meet_precision<BE>(fixture_base2k: usize)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let mut first_factor_roundtrip = true;
    for (index, preset) in all().unwrap().into_iter().enumerate() {
        let preset = preset_with_max_base2k(&preset, fixture_base2k).unwrap();
        let first_factor = preset.plan().coeffs_to_slots().first_factor_on_raise();
        let roundtrip = index == 0 || (first_factor && std::mem::take(&mut first_factor_roundtrip));
        let mut run = BootstrappingPresetRun::<BE>::setup_with_roundtrip(preset, roundtrip);
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

/// Runs the insecure `N = 2^11` S2C-first preset on `BE` and checks its key
/// shapes and output precision. Unlike [`bootstrapping_presets_meet_precision`],
/// it is small enough for a default test run, so the narrow conjugation and
/// SlotsToCoeffs keys the presets derive run end to end. Backends register it
/// when their radix takes the preset's 52 bits: a smaller fixture radix gives
/// every key one digit, hence one shape.
pub fn small_s2c_preset_meets_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let preset = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    let keys = preset.keys_layout();
    let rows = |key: &dyn GGLWEInfos| key.dnum().as_usize();
    let wide_rows = rows(&keys.automorphism_key);
    // The conjugation runs below the raised width, and SlotsToCoeffs on the
    // narrow input: both get keys of fewer rows than the shared one.
    assert!(rows(&keys.rotation_key(-1)) < wide_rows);
    let cyclotomic_order = 2 * preset.n() as i64;
    let (log_n, plan) = (preset.log_n(), preset.plan());
    let c2s = plan.coeffs_to_slots().galois_elements(log_n, cyclotomic_order);
    let s2c_only: Vec<i64> = plan
        .slots_to_coeffs()
        .galois_elements(log_n, cyclotomic_order)
        .into_iter()
        .filter(|p| !c2s.contains(p))
        .collect();
    assert!(!s2c_only.is_empty());
    assert!(s2c_only.iter().all(|&p| rows(&keys.rotation_key(p)) < wide_rows));

    let mut run = BootstrappingPresetRun::<BE>::setup(preset);
    let (re, im) = run.precision();
    println!(
        "PRECISION preset={} re_min={:.2}b re_avg={:.2}b im_min={:.2}b im_avg={:.2}b",
        run.preset().name(),
        re.min_log2_prec,
        re.avg_log2_prec,
        im.min_log2_prec,
        im.avg_log2_prec,
    );
    let advertised = run.preset().log2_precision() as f64;
    assert!(
        re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
        "preset {} advertises {advertised} bits but measured re_min={:.2} im_min={:.2}",
        run.preset().name(),
        re.min_log2_prec,
        im.min_log2_prec
    );
}
/// Scratch for unprepared context generation and key preparation.
/// Query DFT preparation on the unprepared context, then bootstrap execution on the prepared context.
pub(crate) fn bootstrap_setup_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    layout: &CKKSLayout,
    plan: &BootstrappingPlan,
    keys: &BootstrappingKeysLayout,
) -> usize
where
    Module<BE>: CKKSAllOpsTmpBytes<BE>,
{
    let coeffs = plan.eval_mod().coeffs_meta;
    let coeffs = ckks_spec(
        layout.n().as_usize(),
        layout.base2k().as_usize(),
        coeffs.log_delta(),
        coeffs.log_budget(),
    );
    let mut bytes = 0;
    for atk in &keys.rotation_key_layouts() {
        bytes = bytes.max(module.ckks_all_ops_with_atk_tmp_bytes(layout, &keys.tensor_key, atk, &coeffs));
    }
    bytes
}

/// A preset with EvalMod on the conjugate-invariant ring set up end to end and
/// ready to bootstrap repeatedly, as [`BootstrappingPresetRun`] does for the
/// others: `BE` runs the standard stages, `BCI` EvalMod.
pub struct BootstrappingCIPresetRun<BE: Backend, BCI: Backend> {
    preset: BootstrappingCIPreset,
    module: Module<BE>,
    wide: Module<BE>,
    ci: Module<BCI>,
    context: BootstrappingCIContext<BE, BCI, f64>,
    keys: BootstrappingCIKeysPrepared<BE::OwnedBuf, BE, BCI>,
    scratch: ScratchOwned<BE>,
    scratch_ci: ScratchOwned<BCI>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    want_re: Vec<f64>,
    want_im: Vec<f64>,
}

impl<BE, BCI> BootstrappingCIPresetRun<BE, BCI>
where
    BE: TestContextBackend<Ring = Standard> + crate::oep::CKKSEncapsulatedModUpImpl,
    BCI: TestContextBackend<Ring = poulpy_hal::layouts::ConjugateInvariant>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE, BCI>
        + poulpy_core::GLWECIKeyEncryptSk<BE>
        + poulpy_core::layouts::GLWECIKeyPreparedFactory<BE>,
    Module<BCI>: crate::test_suite::helpers::TestContextSharedModule<BCI> + CKKSEncodingOps<BCI, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<BCI>: ScratchOwnedAlloc<BCI>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    poulpy_core::layouts::BackendGLWESecret<BCI>: poulpy_core::layouts::GLWESecretToBackendRef<BE>,
    poulpy_core::layouts::GLWETensorKey<BE::OwnedBuf, BE::ZnxWord>: poulpy_core::layouts::GGLWEToBackendRef<BCI>,
{
    /// Compiles the context on the standard module of degree `N`, the one of
    /// degree `2N` and the CI one of degree `N`, generates the keys, encrypts the
    /// reference vector at the input layout of the preset it extends, and runs
    /// one bootstrap to check the output width and metadata.
    pub fn setup(preset: BootstrappingCIPreset) -> Self {
        let base = preset.base();
        let (n, base2k) = (base.n(), base.base2k());
        let input_layout = base.input_layout();
        let bootstrap_layout = base.bootstrap_layout();
        let module = Module::<BE>::new(n as u64);
        let wide = Module::<BE>::new(2 * n as u64);
        let ci = Module::<BCI>::new(n as u64);

        let scratch_size = bootstrap_setup_tmp_bytes(&module, &bootstrap_layout, base.plan(), base.keys_layout());
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_size.max(1 << 30));
        let mut scratch_ci = ScratchOwned::<BCI>::alloc(1 << 30);
        let context = BootstrappingCIContext::<BE, BCI, f64>::compile(
            &module,
            &wide,
            &ci,
            base2k.into(),
            preset.plan(),
            &mut scratch.borrow(),
            &mut scratch_ci.borrow(),
        )
        .unwrap();
        let mut source_sk = Source::new([0; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, base.dense_secret_hamming_weight(), &mut source_sk);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let keys = context
            .generate_keys(
                &module,
                &wide,
                &ci,
                &sk_raw,
                preset.keys_layout(),
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
                &mut scratch_ci.borrow(),
            )
            .unwrap()
            .prepare::<BE, BCI>(&module, &wide, &ci, &mut scratch.borrow(), &mut scratch_ci.borrow());

        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut input_pt = module.ckks_pt_vec_alloc(module.ckks_max_slots(), base2k.into(), input_layout.k());
        input_pt.set_meta(input_layout.meta());
        module
            .ckks_encode_reim_into(&mut input_pt, &want_re, &want_im, &mut scratch.borrow())
            .unwrap();
        let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        let (mut input_xa, mut input_xe) = (Source::new([3; 32]), Source::new([4; 32]));
        module
            .ckks_encrypt_sk(
                &mut input,
                &input_pt,
                &sk,
                &mut input_xe,
                &mut input_xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        let output = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);

        // The setup takes generous arenas, the compilation and key generation of the
        // CI pipeline having no scratch query; the bootstraps run on exactly the
        // scratch they advertise.
        let (boot_scratch, boot_scratch_ci) =
            module.ckks_bootstrap_ci_tmp_bytes(&wide, &ci, &bootstrap_layout, &input_layout, &context, preset.keys_layout());
        let scratch = ScratchOwned::<BE>::alloc(boot_scratch);
        let scratch_ci = ScratchOwned::<BCI>::alloc(boot_scratch_ci);

        let mut run = Self {
            preset,
            module,
            wide,
            ci,
            context,
            keys,
            scratch,
            scratch_ci,
            input,
            output,
            sk,
            want_re,
            want_im,
        };
        run.bootstrap();
        let base = run.preset.base();
        assert_eq!(
            run.output.k().as_usize(),
            base.output_k(),
            "{}: output width",
            run.preset.name()
        );
        assert_eq!(
            run.output.meta(),
            base.output_layout().meta(),
            "{}: output metadata",
            run.preset.name()
        );
        run
    }

    /// The preset being exercised.
    pub fn preset(&self) -> &BootstrappingCIPreset {
        &self.preset
    }

    /// Bootstraps the reference input into the output ciphertext.
    pub fn bootstrap(&mut self) {
        self.output.set_k(self.preset.bootstrap_k().into());
        self.module
            .ckks_bootstrap_ci(
                &self.wide,
                &self.ci,
                &mut self.output,
                &self.input,
                &self.context,
                &self.keys,
                &mut self.scratch.borrow(),
                &mut self.scratch_ci.borrow(),
            )
            .unwrap();
    }

    /// Bootstraps the reference input at the working width `k` and returns the
    /// error, for a width the keys do not cover.
    pub fn bootstrap_error_at(&mut self, k: usize) -> crate::CKKSError {
        let mut output = self
            .module
            .ckks_ciphertext_alloc(self.preset.base().base2k().into(), k.into());
        self.module
            .ckks_bootstrap_ci(
                &self.wide,
                &self.ci,
                &mut output,
                &self.input,
                &self.context,
                &self.keys,
                &mut self.scratch.borrow(),
                &mut self.scratch_ci.borrow(),
            )
            .unwrap_err()
    }

    /// Decrypts the last bootstrap output and measures its precision against
    /// the reference vector, as `(real, imaginary)` statistics.
    pub fn precision(&mut self) -> (PrecisionStats, PrecisionStats) {
        let output = &self.output;
        assert_canonical_at_k::<BE>("bootstrap CI preset", output);
        let log_budget = output
            .log_budget()
            .min(PRECISION_LOG_BUDGET)
            .min(127usize.saturating_sub(output.log_delta()));
        let mut output_pt = self.module.ckks_pt_vec_alloc(
            self.module.ckks_max_slots(),
            output.base2k(),
            (output.log_delta() + log_budget).into(),
        );
        output_pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(output.log_delta()),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        self.module
            .ckks_decrypt(&mut output_pt, output, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let n = self.preset.base().n();
        let (mut got_re, mut got_im) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        self.module
            .ckks_decode_reim_into(&output_pt, &mut got_re, &mut got_im, &mut self.scratch.borrow())
            .unwrap();
        let log_delta = self.preset.base().log_delta();
        (
            precision_stats(&got_re, &self.want_re, log_delta),
            precision_stats(&got_im, &self.want_im, log_delta),
        )
    }
}

/// Runs every preset with EvalMod on the conjugate-invariant ring
/// ([`all_ci`]) once, with `BE` for the standard stages and `BCI` for
/// EvalMod, and checks the measured output precision against the precision the
/// preset advertises, at the output width and scale of the preset it extends.
/// Full-size bootstraps are slow, so backends register this as an ignored test.
pub fn bootstrapping_ci_presets_meet_precision<BE, BCI>()
where
    BE: TestContextBackend<Ring = Standard> + crate::oep::CKKSEncapsulatedModUpImpl,
    BCI: TestContextBackend<Ring = poulpy_hal::layouts::ConjugateInvariant>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE, BCI>
        + poulpy_core::GLWECIKeyEncryptSk<BE>
        + poulpy_core::layouts::GLWECIKeyPreparedFactory<BE>,
    Module<BCI>: crate::test_suite::helpers::TestContextSharedModule<BCI> + CKKSEncodingOps<BCI, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<BCI>: ScratchOwnedAlloc<BCI>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    poulpy_core::layouts::BackendGLWESecret<BCI>: poulpy_core::layouts::GLWESecretToBackendRef<BE>,
    poulpy_core::layouts::GLWETensorKey<BE::OwnedBuf, BE::ZnxWord>: poulpy_core::layouts::GGLWEToBackendRef<BCI>,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    for preset in all_ci().unwrap() {
        let mut run = BootstrappingCIPresetRun::<BE, BCI>::setup(preset);

        // A working width past what the keys cover fails at entry: 1024 bits is more
        // than any key keeps beyond the width it was laid out for.
        let err = run.bootstrap_error_at(run.preset().bootstrap_k() + 1024);
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MissingAutomorphismKey {
                    op: "ckks_bootstrap_ci",
                    ..
                })
            ),
            "{}: expected a key refused at entry, got: {err}",
            run.preset().name()
        );

        let (re, im) = run.precision();
        let base = run.preset().base();
        println!(
            "PRECISION backend={backend} preset={} base2k={} re_min={:.2}b re_avg={:.2}b im_min={:.2}b im_avg={:.2}b (advertised {}b)",
            run.preset().name(),
            base.base2k(),
            re.min_log2_prec,
            re.avg_log2_prec,
            im.min_log2_prec,
            im.avg_log2_prec,
            base.log2_precision(),
        );
        let advertised = base.log2_precision() as f64;
        assert!(
            re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
            "preset {} advertises {advertised} bits but measured re_min={:.2} im_min={:.2}",
            run.preset().name(),
            re.min_log2_prec,
            im.min_log2_prec
        );
    }
}
