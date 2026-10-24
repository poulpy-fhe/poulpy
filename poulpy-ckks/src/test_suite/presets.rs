//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.

use crate::Scale;
use poulpy_core::{
    GLWECIKeyEncryptSk,
    api::GLWECIKeyPreparedFactory,
    layouts::{
        GGLWEInfos, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPrepared, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, SetK, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ReaderFrom, ScratchArena, ScratchOwned, Standard, WriterTo},
    source::Source,
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDFTOps, CKKSDecryptOps,
        CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingPlan, CKKSCiphertextOwned, CKKSModuleAlloc,
        CKKSPlaintextOwned, EvalModRing,
    },
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::helpers::{
        PrecisionStats, TestContextBackend, TestContextModule, TestContextSharedModule, assert_canonical_at_k, ckks_spec,
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

/// The conjugate-invariant twin of `BE` that runs EvalMod in the bootstrap of
/// [`CKKSBootstrappingCIOps`].
pub type CIBackend<BE> = <Module<BE> as CKKSBootstrappingCIOps<BE>>::CI;

/// The bootstrap of a preset run: its context and keys, and those of the CI ring.
type Bootstrap<BE> = Box<
    dyn FnMut(&Module<BE>, &mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> CKKSResult<()>,
>;

/// A preset set up end to end and ready to bootstrap repeatedly.
pub struct BootstrappingPresetRun<BE: Backend> {
    preset: BootstrappingPreset,
    module: Module<BE>,
    bootstrap: Bootstrap<BE>,
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
            Self::roundtrip_keys(&module, &keys);
        }
        let keys = keys.prepare(&module, &mut scratch.borrow());

        let bootstrap: Bootstrap<BE> =
            Box::new(move |module, output, input, scratch| module.ckks_bootstrap(output, input, &context, &keys, scratch));
        Self::with_bootstrap(preset, module, scratch, sk, bootstrap)
    }

    /// Writes every key of `keys` and reads it back into a fresh key of its layout, which must
    /// equal it.
    fn roundtrip_keys(module: &Module<BE>, keys: &crate::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>) {
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
        if let Some((trace, embed)) = &keys.conversion_keys {
            roundtrip(trace, module.glwe_ci_trace_key_alloc_from_infos(trace));
            roundtrip(embed, module.glwe_ci_embed_key_alloc_from_infos(embed));
        }
    }

    /// [`Self::setup`] for a preset with EvalMod on the conjugate-invariant ring, on the twin [`CIBackend`].
    pub fn setup_ci(preset: BootstrappingPreset) -> Self
    where
        CIBackend<BE>: TestContextBackend,
        Module<BE>: CKKSBootstrappingCIOps<BE> + GLWECIKeyEncryptSk<BE> + GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    {
        Self::setup_ci_with_roundtrip(preset, false)
    }

    /// [`Self::setup_ci`], with every key round-tripped through its serialization when
    /// `roundtrip_keys`.
    fn setup_ci_with_roundtrip(preset: BootstrappingPreset, roundtrip_keys: bool) -> Self
    where
        CIBackend<BE>: TestContextBackend,
        Module<BE>: CKKSBootstrappingCIOps<BE> + GLWECIKeyEncryptSk<BE> + GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    {
        // One module of degree 2N serves the conversions and the stages of degree N.
        let n = preset.n() as u64;
        let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
        let sk_layout = preset.bootstrap_layout().glwe_layout;
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&sk_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);

        // Every step runs on arenas of exactly its own query.
        let context = compile_ci(&module, &ci, &preset);
        let keys = generate_keys_ci(&module, &ci, &context, &sk_raw, preset.keys_layout());
        if roundtrip_keys {
            Self::roundtrip_keys(&module, &keys);
        }
        let keys = prepare_keys_ci(&module, &ci, &keys, preset.keys_layout());
        let (layout, input_layout) = (preset.bootstrap_layout(), preset.input_layout());
        let (bytes, bytes_ci) = module.ckks_bootstrap_ci_tmp_bytes(&ci, &layout, &input_layout, &context, preset.keys_layout());
        let mut scratch_ci = ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci);
        let bootstrap: Bootstrap<BE> = Box::new(move |module, output, input, scratch| {
            module.ckks_bootstrap_ci_into(&ci, output, input, &context, &keys, scratch, &mut scratch_ci.borrow())
        });
        Self::with_bootstrap(preset, module, ScratchOwned::alloc(bytes), sk, bootstrap)
    }

    /// Encrypts the reference vector at the preset's input layout and runs one
    /// bootstrap to check the output width.
    fn with_bootstrap(
        preset: BootstrappingPreset,
        module: Module<BE>,
        mut scratch: ScratchOwned<BE>,
        sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
        bootstrap: Bootstrap<BE>,
    ) -> Self {
        let (n, base2k) = (preset.n(), preset.base2k());
        let (input_layout, bootstrap_layout) = (preset.input_layout(), preset.bootstrap_layout());
        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut input_pt = module.ckks_pt_vec_alloc(n / 2, base2k.into(), input_layout.k());
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
            bootstrap,
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
        (self.bootstrap)(&self.module, &mut self.output, &self.input, &mut self.scratch.borrow()).unwrap();
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
        let mut output_pt =
            self.module
                .ckks_pt_vec_alloc(self.preset.n() / 2, output.base2k(), (output.log_delta() + log_budget).into());
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

    /// Checks that a working width past what the keys cover fails at entry, and
    /// that the measured precision reaches the one the preset advertises.
    fn check(&mut self, backend: &str) {
        // 1024 bits is more than any key keeps beyond the width it was laid out for.
        let k = self.preset.bootstrap_k() + 1024;
        let mut layout = self.preset.bootstrap_layout();
        layout.glwe_layout.k = k.into();
        let mut output = self.module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        let err = (self.bootstrap)(&self.module, &mut output, &self.input, &mut self.scratch.borrow()).unwrap_err();
        assert!(
            matches!(err.composition(), Some(CKKSCompositionError::MissingAutomorphismKey { .. })),
            "{}: expected a key refused at entry, got: {err}",
            self.preset.name()
        );
        let (re, im) = self.precision();
        let preset = &self.preset;
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

/// Runs every preset once on `BE` and checks the measured output precision
/// against the precision the preset advertises, and that its keys are refused
/// at entry past their width. The twin [`CIBackend`] runs EvalMod for the presets that put it on
/// the conjugate-invariant ring. Every preset runs at the fixture radix as
/// [`preset_with_max_base2k`] adapts it. The first preset also round-trips every
/// generated key, both encapsulation directions included.
///
/// The caller supplies the fixture radix limit; this precision check does
/// not establish a failure-probability bound. Every backend uses `f64` DFT
/// matrices and must reach the advertised precision. Full-size bootstraps are
/// slow, so backends register this as an ignored test.
pub fn bootstrapping_presets_meet_precision<BE>(fixture_base2k: usize)
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    // The keys of the first preset of each ring go through their serialization.
    let mut first_ci = true;
    for (index, preset) in all().unwrap().into_iter().enumerate() {
        let ring = preset.plan().eval_mod_ring();
        let preset = preset_with_max_base2k(&preset, fixture_base2k).unwrap();
        match ring {
            EvalModRing::Standard => BootstrappingPresetRun::<BE>::setup_with_roundtrip(preset, index == 0).check(backend),
            _ => BootstrappingPresetRun::<BE>::setup_ci_with_roundtrip(preset, std::mem::take(&mut first_ci)).check(backend),
        }
    }
}

/// Runs the insecure `N = 2^11` S2C-first preset on `BE` and checks its key
/// shapes and output precision, then the same preset with EvalMod on the
/// conjugate-invariant ring, on the twin [`CIBackend`]. Unlike [`bootstrapping_presets_meet_precision`],
/// it is small enough for a default test run, so the narrow conjugation and
/// SlotsToCoeffs keys the presets derive run end to end. Backends register it
/// when their radix takes the preset's 52 bits: a smaller fixture radix gives
/// every key one digit, hence one shape.
pub fn small_s2c_preset_meets_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
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

    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    BootstrappingPresetRun::<BE>::setup(preset.clone()).check(backend);
    let ci = EvalModRing::ConjugateInvariant {
        pre_log_delta: 46,
        post_log_delta: 38,
    };
    // Its keys, conversion keys included, go through their serialization.
    BootstrappingPresetRun::<BE>::setup_ci_with_roundtrip(preset.with_eval_mod_ring(ci).unwrap(), true).check(backend);
}

/// The bootstrap with EvalMod on the conjugate-invariant ring refuses, before any
/// work: conversion keys narrower than its widths, a context compiled without
/// conversions, modules of the wrong degrees, and keys without the
/// dense-to-sparse key of an encapsulated recipe. On the insecure `N = 2^11`
/// preset with EvalMod on the CI ring.
pub fn ci_bootstrap_refuses_invalid_inputs<BE>()
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let standard = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    let ring = EvalModRing::ConjugateInvariant {
        pre_log_delta: 46,
        post_log_delta: 38,
    };
    let preset = standard.with_eval_mod_ring(ring).unwrap();
    let n = preset.n() as u64;
    let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
    let sk_layout = preset.bootstrap_layout().glwe_layout;
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
    let context = compile_ci(&module, &ci, &preset);
    let compiled = {
        let mut scratch = ScratchOwned::<BE>::alloc(bootstrap_setup_tmp_bytes(
            &module,
            &standard.bootstrap_layout(),
            standard.plan(),
            standard.keys_layout(),
        ));
        BootstrappingContext::<BE, f64>::compile(&module, preset.base2k().into(), standard.plan(), &mut scratch.borrow()).unwrap()
    };
    // Key generation leaves nothing of the CI secret, nor of the encryptions, in the CI scratch.
    let (bytes, bytes_ci) = preset.keys_layout().generate_ci_tmp_bytes(&module, &ci);
    poulpy_core::test_suite::assert_wipes_scratch::<CIBackend<BE>>(bytes_ci, |scratch_ci| {
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        context
            .generate_keys_ci(
                &module,
                &ci,
                &sk_raw,
                preset.keys_layout(),
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
                scratch_ci,
            )
            .unwrap();
    });
    let keys_for = |layout: &BootstrappingKeysLayout| {
        prepare_keys_ci(
            &module,
            &ci,
            &generate_keys_ci(&module, &ci, &context, &sk_raw, layout),
            layout,
        )
    };
    let keys = keys_for(preset.keys_layout());
    // A trace key, then an embed key, one digit narrower than the bootstrap needs.
    let narrow_keys: Vec<_> = (0..2)
        .map(|which| {
            let mut narrow = preset.keys_layout().clone();
            let (trace, embed) = narrow.conversion_keys.as_mut().unwrap();
            let key = if which == 0 { trace } else { embed };
            key.dnum = (key.dnum.as_usize() - 1).into();
            keys_for(&narrow)
        })
        .collect();
    let mut no_dense_to_sparse = keys_for(preset.keys_layout());
    no_dense_to_sparse.encapsulation_keys = None;

    let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&preset.input_layout());
    input.set_meta(preset.input_layout().meta());
    let mut output = module.ckks_ciphertext_alloc_from_glwe_infos(&preset.bootstrap_layout());
    // The bootstrap runs on exactly its query, and refuses either arena one alignment unit short.
    let (bytes, bytes_ci) = module.ckks_bootstrap_ci_tmp_bytes(
        &ci,
        &preset.bootstrap_layout(),
        &preset.input_layout(),
        &context,
        preset.keys_layout(),
    );
    let mut bootstrap_on = |module: &Module<BE>,
                            context: &BootstrappingContext<BE, f64>,
                            keys: &crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE, CIBackend<BE>>,
                            (bytes, bytes_ci): (usize, usize)| {
        let (mut scratch, mut scratch_ci) = (
            ScratchOwned::<BE>::alloc(bytes),
            ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci),
        );
        output.set_k(preset.bootstrap_k().into());
        let before = output.to_host_owned::<BE>();
        let result = module.ckks_bootstrap_ci_into(
            &ci,
            &mut output,
            &input,
            context,
            keys,
            &mut scratch.borrow(),
            &mut scratch_ci.borrow(),
        );
        // A refusal leaves the output as it was.
        if result.is_err() {
            let after = output.to_host_owned::<BE>();
            assert!(
                (after.meta(), after.k(), after.data()) == (before.meta(), before.k(), before.data()),
                "a refusal wrote the output"
            );
        }
        result
    };
    for short in [
        (bytes - BE::SCRATCH_ALIGN, bytes_ci),
        (bytes, bytes_ci - CIBackend::<BE>::SCRATCH_ALIGN),
    ] {
        let err = bootstrap_on(&module, &context, &keys, short).unwrap_err();
        assert!(matches!(err, crate::CKKSError::Internal(_)), "short arena: {err}");
    }
    let mut bootstrap =
        |module: &Module<BE>,
         context: &BootstrappingContext<BE, f64>,
         keys: &crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE, CIBackend<BE>>| {
            bootstrap_on(module, context, keys, (bytes, bytes_ci))
        };
    bootstrap(&module, &context, &keys).unwrap();
    for (err, cause) in [
        (bootstrap(&module, &context, &narrow_keys[0]), "narrow trace key"),
        (bootstrap(&module, &context, &narrow_keys[1]), "narrow embed key"),
        (bootstrap(&module, &compiled, &keys), "context of compile"),
        (bootstrap(&Module::<BE>::new(n), &context, &keys), "module of degree N"),
        (bootstrap(&module, &context, &no_dense_to_sparse), "encapsulation"),
    ] {
        let err = err.unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand {
                    op: "ckks_bootstrap_ci_into",
                    ..
                })
            ),
            "{cause}: {err}"
        );
    }
}

/// [`BootstrappingContext::compile_ci`] of `preset`, each stage on arenas of exactly its query.
fn compile_ci<BE>(module: &Module<BE>, ci: &Module<CIBackend<BE>>, preset: &BootstrappingPreset) -> BootstrappingContext<BE, f64>
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE>
        + CKKSDFTOps<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
{
    let (bytes, bytes_ci) = BootstrappingContext::<BE, f64>::compile_ci_unprepared_tmp_bytes(module, ci);
    let unprepared = BootstrappingContext::<BE, f64>::compile_ci_unprepared(
        module,
        ci,
        preset.base2k().into(),
        preset.plan(),
        &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        &mut ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci).borrow(),
    )
    .unwrap();
    let bytes = unprepared.prepare_tmp_bytes(module);
    unprepared.prepare(module, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
}

/// [`BootstrappingContext::generate_keys_ci`] of `layout`, on arenas of exactly its query.
fn generate_keys_ci<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    context: &BootstrappingContext<BE, f64>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    layout: &BootstrappingKeysLayout,
) -> crate::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSBootstrappingCIOps<BE> + GLWECIKeyEncryptSk<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>,
{
    let (bytes, bytes_ci) = layout.generate_ci_tmp_bytes(module, ci);
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    context
        .generate_keys_ci(
            module,
            ci,
            sk_raw,
            layout,
            &mut source_xs,
            &mut source_xe,
            &mut source_xa,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
            &mut ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci).borrow(),
        )
        .unwrap()
}

/// [`BootstrappingKeySet::prepare_ci`](crate::layouts::BootstrappingKeySet::prepare_ci) of `keys`, generated for
/// `layout`, on arenas of exactly its query.
fn prepare_keys_ci<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    keys: &crate::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>,
    layout: &BootstrappingKeysLayout,
) -> crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE, CIBackend<BE>>
where
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSBootstrappingCIOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>,
{
    let (bytes, bytes_ci) = layout.prepare_ci_tmp_bytes(module, ci);
    keys.prepare_ci(
        module,
        ci,
        &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        &mut ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci).borrow(),
    )
    .unwrap()
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
