//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.

use crate::Scale;
use poulpy_core::{
    GLWECIKeyEncryptSk, GLWEDiagonalProd, GLWELinearTransformationBabyStepsBytesOf, GLWERotate,
    ScratchArenaTakeGLWELinearTransformationBabySteps,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWECIKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory, TransferInto,
    },
    layouts::{
        BackendGLWESecret, BackendGLWESecretPrepared, Base2K, GGLWEInfos, GLWEAutomorphismKeyAlloc, GLWECIEmbedKeyAlloc,
        GLWECITraceKeyAlloc, GLWESecretAlloc, GLWESecretPrepared, GLWESwitchingKeyAlloc, GLWETensorKeyAlloc,
        GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, SetGLWEPlaintextInfos, SetK,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostBytesBackend, HostDataMut, HostDataRef, HostStaged, MaxBase2k, Module, ReaderFrom, ScratchArena,
        ScratchOwned, Standard, WriterTo,
    },
    source::Source,
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos,
    SlotsKind,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory,
        CKKSDFTOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSEncryptOps, CKKSEvalModOps,
        CKKSLinearTransformationOps, CKKSPow2Ops,
    },
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeysLayout, BootstrappingKeysPrepared, BootstrappingLinearKeys,
        BootstrappingPipeline, BootstrappingPlan, CIConversion, CKKSCITwin, CKKSCiphertextAlloc, CKKSCiphertextOwned,
        CKKSDFTFactors, CKKSModRaiseInfos, CKKSPlaintextAlloc, CKKSPlaintextOwned, CKKSRightOperandPrepared, CoeffsMaskPrepared,
        DFTOutputFormat, DFTPlan, DFTType, Decode, Encode, EncodedLut, EvalModRing, ModRaisePlan, ModRaisePrepared, Split,
    },
    oep::CKKSModUpFirstFactorImpl,
    polynomial::SplitStrategy,
    presets::bootstrapping::{BootstrappingPreset, PresetScales, all},
    test_suite::bootstrapping::{assert_eval_mod_refuses, assert_mod_raise_refuses},
    test_suite::helpers::{
        LoggedBootstrappingKeys, PrecisionStats, TestContextBackend, TestContextModule, TestContextSharedModule,
        assert_canonical_at_k, assert_invalid_operand, ckks_spec, precision_stats, test_vector_1,
    },
    test_suite::parity::helpers::{fixture_ciphertext, poisoned_scratch, snapshot, with_scratch},
};

/// Plaintext budget bits (above `log_delta`) used to measure the output precision.
pub const PRECISION_LOG_BUDGET: usize = 8;

/// Failure probability, `2^-BOOTSTRAP_FAILURE_BITS` per bootstrap, the test and
/// benchmark radix is selected for (see [`preset_for_backend`]).
pub const BOOTSTRAP_FAILURE_BITS: usize = 128;

/// The preset at the largest radix backend `BE`, and `BCI` for EvalMod on the
/// conjugate-invariant ring, take for a bootstrap failure probability of
/// `2^-BOOTSTRAP_FAILURE_BITS` ([`BootstrappingPreset::max_backend_base2k`]):
/// above the nominal radix the keys are re-derived as the preset derives them,
/// below it as [`preset_with_max_base2k`] adapts them.
pub fn preset_for_backend<BE: MaxBase2k, BCI: MaxBase2k>(preset: &BootstrappingPreset) -> anyhow::Result<BootstrappingPreset> {
    let max = preset.max_backend_base2k::<BE, BCI>(BOOTSTRAP_FAILURE_BITS);
    if max > preset.base2k() {
        preset.with_backend_base2k::<BE, BCI>(BOOTSTRAP_FAILURE_BITS)
    } else {
        preset_with_max_base2k(preset, max)
    }
}

/// Caps the nominal preset at the caller's explicit test or benchmark radix.
///
/// This fixture choice carries no failure-probability guarantee. A smaller
/// radix caps the high-modulus digits at 7 limbs, each key taking the digit of
/// fewest rows its use admits under the cap, and gives the dense-to-sparse key
/// 1 limb. The circuit and bit widths stay unchanged.
pub fn preset_with_max_base2k(preset: &BootstrappingPreset, fixture_base2k: usize) -> anyhow::Result<BootstrappingPreset> {
    if preset.base2k() <= fixture_base2k {
        Ok(preset.clone())
    } else {
        preset.with_base2k(fixture_base2k)?.with_dsizes(7, 1)
    }
}

/// The conjugate-invariant twin of `BE` that runs EvalMod in a bootstrap of a plan on that ring.
pub type CIBackend<BE> = <BE as CKKSCITwin>::CI;

/// `Module<BE>` capabilities of the backend a [`BootstrappingPresetRun`] evaluates on.
pub trait PresetRunModule<BE: CKKSCITwin>:
    ModuleNew<BE>
    + ModuleSynchronize<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + CKKSAllOpsTmpBytes<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSBootstrappingOps<BE>
    + CKKSDFTOps<BE>
    + CKKSEvalModOps<BE>
    + CKKSDFTMatrixOps<BE, f64>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + GLWESwitchingKeyPreparedFactory<BE>
    + crate::api::CKKSRightOperandPreparedFactory<BE>
{
}

impl<BE: CKKSCITwin, M> PresetRunModule<BE> for M where
    M: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
{
}

/// The bootstrap of a preset run: its context and keys, and those of the CI ring. It returns how many times its
/// stages looked the tensor key up.
type Bootstrap<BE> = Box<
    dyn FnMut(
        &Module<BE>,
        &mut CKKSCiphertextOwned<BE>,
        &CKKSCiphertextOwned<BE>,
        &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<usize>,
>;

/// A preset set up end to end and ready to bootstrap repeatedly.
///
/// `BE` evaluates: it compiles the context, prepares the keys and bootstraps.
/// `BC` is the client: it samples the secret, generates the keys from the plan,
/// encrypts the reference vector and decrypts the result. The client needs host
/// storage; the evaluated backend only exchanges coefficient data with the host.
/// A host backend plays both roles, which is the default.
pub struct BootstrappingPresetRun<BE: Backend, BC: Backend = BE> {
    preset: BootstrappingPreset,
    module: Module<BE>,
    client: Module<BC>,
    bootstrap: Bootstrap<BE>,
    scratch: ScratchOwned<BE>,
    client_scratch: ScratchOwned<BC>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    sk: GLWESecretPrepared<BC::OwnedBuf, BC>,
    want_re: Vec<f64>,
    want_im: Vec<f64>,
    /// The tensor-key lookups of the stages of the last bootstrap.
    tensor_lookups: usize,
}

impl<BE, BC> BootstrappingPresetRun<BE, BC>
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + 'static + CKKSCITwin,
    BC: TestContextBackend<Ring = Standard> + CKKSCITwin,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE> + PresetRunModule<BE>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    /// Compiles the context, generates the keys on the client and moves them to
    /// the evaluated backend, encrypts the reference vector at the preset's input
    /// layout, and runs one bootstrap to check the output width. Every layout
    /// comes from `preset`. With EvalMod on the conjugate-invariant ring, the run
    /// takes `ci`, of the preset's degree, and the client its own twin [`CIBackend`]:
    /// the bootstraps of the run keep reading `ci`.
    pub fn setup(preset: BootstrappingPreset, ci: Option<Module<CIBackend<BE>>>) -> Self
    where
        Module<BE>: GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSEncodingOps<CIBackend<BE>, f64>
            + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
        CIBackend<BC>: TestContextBackend,
        Module<BC>: GLWECIKeyEncryptSk<BC>,
        Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    {
        Self::setup_with_roundtrip(preset, ci, false)
    }

    /// [`Self::setup`], with every key round-tripped through its serialization when
    /// `roundtrip_keys`.
    fn setup_with_roundtrip(preset: BootstrappingPreset, ci: Option<Module<CIBackend<BE>>>, roundtrip_keys: bool) -> Self
    where
        Module<BE>: GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSEncodingOps<CIBackend<BE>, f64>
            + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
        CIBackend<BC>: TestContextBackend,
        Module<BC>: GLWECIKeyEncryptSk<BC>,
        Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    {
        // On the CI ring, one module of degree 2N serves the conversions and the stages of degree N.
        let n = preset.n() as u64;
        let degree = if ci.is_some() { 2 * n } else { n };
        let (module, client) = (Module::<BE>::new(degree), Module::<BC>::new(degree));
        let client_ci = ci.as_ref().map(|_| Module::<CIBackend<BC>>::new(n));
        let (sk_raw, sk) = dense_secret(&client, &preset);

        // Every step runs on an arena of exactly its own query.
        let (plan, keys_layout, base2k) = (preset.plan(), preset.keys_layout(), preset.base2k());
        let bytes = keys_layout.generate_tmp_bytes(&client, client_ci.as_ref());
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let keys = plan
            .generate_keys(
                &client,
                client_ci.as_ref(),
                &sk_raw,
                keys_layout,
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut ScratchOwned::<BC>::alloc(bytes).borrow(),
            )
            .unwrap();
        if roundtrip_keys {
            Self::roundtrip_keys(&client, &keys);
        }

        let context = compile_context::<BE, f64>(&module, ci.as_ref(), base2k.into(), plan);
        let bytes = keys_layout.prepare_tmp_bytes(&module, ci.as_ref());
        let keys = keys
            .transfer_and_prepare(&module, ci.as_ref(), &mut ScratchOwned::<BE>::alloc(bytes).borrow())
            .unwrap();

        let (layout, input_layout) = (preset.bootstrap_layout(), preset.input_layout());
        let client_scratch = ScratchOwned::<BC>::alloc(bootstrap_setup_tmp_bytes(&client, &layout, plan, keys_layout));
        let (plan, log_n, label) = (plan.clone(), preset.log_n(), preset_label(&preset));
        let bytes = module.ckks_bootstrap_tmp_bytes(&layout, &input_layout, &context, ci.as_ref(), keys_layout);
        let bootstrap: Bootstrap<BE> = Box::new(move |module, output, input, scratch| {
            // The bootstrap lowers the output's width to its result.
            let widths = plan.key_widths(log_n, output.k().as_usize(), input.k().as_usize(), input.log_delta());
            let logged = LoggedBootstrappingKeys::new::<BE>(&keys);
            module.ckks_bootstrap(output, input, &context, ci.as_ref(), &logged, scratch)?;
            Ok(logged.assert_lookups_at(&widths, &label))
        });
        Self::with_bootstrap(
            preset,
            module,
            poisoned_scratch::<BE>(bytes),
            client,
            client_scratch,
            sk,
            bootstrap,
        )
    }

    /// Writes every key of `keys` and reads it back into a fresh key of its layout, which must
    /// equal it.
    fn roundtrip_keys(module: &Module<BC>, keys: &crate::layouts::BootstrappingKeySet<BC::OwnedBuf, BC::ZnxWord>) {
        fn roundtrip<T: ReaderFrom + WriterTo + PartialEq>(key: &T, mut restored: T) {
            let mut bytes = Vec::new();
            key.write_to(&mut bytes).unwrap();
            restored.read_from(&mut bytes.as_slice()).unwrap();
            assert!(restored == *key);
        }
        for key in keys.rotation_keys.values() {
            roundtrip(key, module.glwe_automorphism_key_alloc_from_infos(key));
        }
        for key in keys.tensor_key.iter().chain(&keys.ci_tensor_key) {
            roundtrip(key, module.glwe_tensor_key_alloc_from_infos(key));
        }
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

    /// Encrypts the reference vector at the preset's input layout on the client,
    /// moves it to the evaluated backend, and runs one bootstrap to check the
    /// output width.
    fn with_bootstrap(
        preset: BootstrappingPreset,
        module: Module<BE>,
        scratch: ScratchOwned<BE>,
        client: Module<BC>,
        client_scratch: ScratchOwned<BC>,
        sk: GLWESecretPrepared<BC::OwnedBuf, BC>,
        bootstrap: Bootstrap<BE>,
    ) -> Self {
        let (input_layout, bootstrap_layout) = (preset.input_layout(), preset.bootstrap_layout());
        let (want_re, want_im) = test_vector_1::<f64>(preset.n() / 2);
        let input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        let output = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);

        let mut run = Self {
            preset,
            module,
            client,
            bootstrap,
            scratch,
            client_scratch,
            input,
            output,
            sk,
            want_re,
            want_im,
            tensor_lookups: 0,
        };
        run.encrypt_input(SlotsKind::Complex);
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

    /// Encrypts the reference vector into the input, at the preset's input layout
    /// with slots of kind `slots`, on the client, and moves it to the evaluated backend.
    fn encrypt_input(&mut self, slots: SlotsKind) {
        let (n, base2k, input_layout) = (self.preset.n(), self.preset.base2k(), self.preset.input_layout());
        let client = &self.client;
        let mut input_pt = client.ckks_plaintext_alloc_from_slots(n / 2, base2k.into(), input_layout.k());
        input_pt.set_meta(GLWEPlaintextMeta {
            slots,
            ..input_layout.meta()
        });
        client
            .ckks_encode_slots_host_into(&mut input_pt, &self.want_re, &self.want_im, &mut self.client_scratch.borrow())
            .unwrap();

        let mut client_input = client.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        let (mut input_xa, mut input_xe) = (Source::new([3; 32]), Source::new([4; 32]));
        client
            .ckks_encrypt_sk(
                &mut client_input,
                &input_pt,
                &self.sk,
                &mut input_xe,
                &mut input_xa,
                &mut self.client_scratch.borrow(),
            )
            .unwrap();
        client_input.inner.transfer_into(&mut self.input.inner);
        self.input.set_meta(client_input.meta());
    }

    /// The run on real slots: the real part of the reference vector, an imaginary
    /// part of zero, in an input of [`SlotsKind::Real`] slots, which the S2C-first
    /// bootstrap evaluates with one EvalMod, where complex slots take two. Bootstraps it
    /// once to check the output and the path.
    pub fn with_real_slots(mut self) -> Self {
        let complex = self.tensor_lookups;
        self.want_im.fill(0.0);
        self.encrypt_input(SlotsKind::Real);
        self.bootstrap();
        assert!(complex > 0, "the complex run relinearized nothing");
        // The step resolves its tensor key once on either route, before EvalMod's own lookups.
        assert_eq!(
            2 * (self.tensor_lookups - 1),
            complex - 1,
            "real slots did not take the one-EvalMod path"
        );
        assert_eq!(self.output.k().as_usize(), self.preset.output_k());
        assert_eq!(self.output.meta().slots, SlotsKind::Real);
        self
    }

    /// The preset being exercised.
    pub fn preset(&self) -> &BootstrappingPreset {
        &self.preset
    }

    /// Bootstraps the reference input into the output ciphertext and returns
    /// once the evaluated backend has completed it.
    pub fn bootstrap(&mut self) {
        self.bootstrap_queued();
        self.synchronize();
    }

    /// [`Self::bootstrap`] without waiting for the evaluated backend: on a
    /// backend that queues its work, the bootstrap may still be running when
    /// this returns. [`Self::synchronize`] waits for it.
    pub fn bootstrap_queued(&mut self) {
        self.output.set_k(self.preset.bootstrap_k().into());
        self.tensor_lookups = (self.bootstrap)(&self.module, &mut self.output, &self.input, &mut self.scratch.borrow()).unwrap();
    }

    /// Returns once the evaluated backend has completed the work queued so far.
    pub fn synchronize(&self) {
        self.module.synchronize();
    }

    /// Moves the last bootstrap output to the client, decrypts it and measures
    /// its precision against the reference vector, as `(real, imaginary)`
    /// statistics.
    pub fn precision(&mut self) -> (PrecisionStats, PrecisionStats) {
        // The allocation keeps the bootstrap width; the bootstrap narrowed `k`.
        let mut output = self
            .client
            .ckks_ciphertext_alloc_from_glwe_infos(&self.preset.bootstrap_layout());
        output.set_k(self.output.k());
        self.output.inner.transfer_into(&mut output.inner);
        output.set_meta(self.output.meta());
        assert_canonical_at_k::<BC>("bootstrap preset", &output);
        let log_budget = output
            .log_budget()
            .min(PRECISION_LOG_BUDGET)
            .min(127usize.saturating_sub(output.log_delta()));
        let mut output_pt = self.client.ckks_plaintext_alloc_from_slots(
            self.preset.n() / 2,
            output.base2k(),
            (output.log_delta() + log_budget).into(),
        );
        output_pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(output.log_delta()),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        self.client
            .ckks_decrypt(&mut output_pt, &output, &self.sk, &mut self.client_scratch.borrow())
            .unwrap();
        let n = self.preset.n();
        let (mut got_re, mut got_im) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        self.client
            .ckks_decode_slots_host_into(&output_pt, &mut got_re, &mut got_im, &mut self.client_scratch.borrow())
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
            preset_label(&self.preset)
        );
        let (re, im) = self.precision();
        let preset = &self.preset;
        println!(
            "PRECISION backend={backend} preset={} base2k={} re_min={:.2}b re_avg={:.2}b im_min={:.2}b im_avg={:.2}b (advertised {}b)",
            preset_label(preset),
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
            preset_label(preset),
            re.min_log2_prec,
            im.min_log2_prec
        );
    }
}

/// Runs every preset once on `BE`, and once more with the keys of power-of-two
/// rotations only ([`BootstrappingPreset::with_power_of_two_keys`]), and
/// checks the measured output precision
/// against the precision the preset advertises, and that its keys are refused
/// at entry past their width. The twin [`CIBackend`] runs EvalMod for the presets that put it on
/// the conjugate-invariant ring. The first preset also round-trips every
/// generated key, both encapsulation directions included.
///
/// Each preset runs at the radix [`preset_for_backend`] selects for `BE`. Every
/// backend uses `f64` DFT matrices and must reach the advertised precision.
/// Full-size bootstraps are slow, so backends register this as an ignored test.
pub fn bootstrapping_presets_meet_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k + CKKSCITwin,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    bootstrapping_presets_meet_precision_with_client::<BE, BE>()
}

/// [`bootstrapping_presets_meet_precision`] for an evaluated backend `BE` whose
/// keys and ciphertexts come from the client backend `BC`, each with its
/// conjugate-invariant twin [`CIBackend`].
pub fn bootstrapping_presets_meet_precision_with_client<BE, BC>()
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static + CKKSCITwin,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE> + PresetRunModule<BE> + CKKSDFTOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::GLWEDiagonalProd<BE>,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    // The keys of the first preset of each ring go through their serialization.
    let mut first_ci = true;
    for (index, nominal) in all().unwrap().into_iter().enumerate() {
        let ring = nominal.plan().eval_mod_ring();
        // Each preset, then its variant with the keys of power-of-two rotations only.
        for preset in [nominal.clone(), nominal.with_power_of_two_keys().unwrap()] {
            let roundtrip = index == 0 && !preset.plan().coeffs_to_slots().power_of_two_keys();
            let preset = preset_for_backend::<BE, CIBackend<BE>>(&preset).unwrap();
            let mut run = match ring {
                EvalModRing::Standard => BootstrappingPresetRun::<BE, BC>::setup_with_roundtrip(preset, None, roundtrip),
                _ => {
                    let ci = Module::<CIBackend<BE>>::new(preset.n() as u64);
                    BootstrappingPresetRun::<BE, BC>::setup_with_roundtrip(preset, Some(ci), std::mem::take(&mut first_ci))
                }
            };
            run.check(backend);
        }
    }
}

/// The preset's name, marked with the variants that keep it: `+erp` with
/// EvalRound+ ([`BootstrappingPreset::with_eval_round_plus`]), `+fold_u` with
/// `u` folded into CoeffsToSlots ([`CIConversion::Folded`]), `+pow2` with the
/// keys of power-of-two rotations only ([`BootstrappingPreset::with_power_of_two_keys`]).
pub fn preset_label(preset: &BootstrappingPreset) -> String {
    let mut label = preset.name().to_string();
    if preset.plan().coeffs_to_slots_bypass().is_some() {
        label.push_str("+erp");
    }
    if let EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        ..
    } = preset.plan().eval_mod_ring()
    {
        label.push_str("+fold_u");
    }
    if preset.plan().coeffs_to_slots().power_of_two_keys() {
        label.push_str("+pow2");
    }
    label
}

/// Runs the insecure `N = 2^11` S2C-first preset on `BE` and checks its key
/// shapes and output precision, on complex and on real slots, then the same
/// preset with EvalRound+, with EvalMod on the conjugate-invariant ring, on the
/// twin [`CIBackend`], and with the keys of power-of-two rotations. Unlike
/// [`bootstrapping_presets_meet_precision`], it is small enough for a default
/// test run, so the narrow conjugation and SlotsToCoeffs keys the presets derive
/// run end to end. Backends register it when their radix takes the preset's 52 bits.
pub fn small_s2c_preset_meets_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k + CKKSCITwin,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let preset = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    let keys = preset.keys_layout();
    let rows = |key: &dyn GGLWEInfos| key.dnum().as_usize();
    let wide_rows = keys.rotation_keys.values().map(|key| rows(key)).max().unwrap();
    // The conjugation runs below the raised width, and SlotsToCoeffs on the
    // narrow input: both get keys of fewer rows than those of the raised width.
    assert!(rows(&keys.rotation_keys[&-1]) < wide_rows);
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
    assert!(s2c_only.iter().all(|&p| rows(&keys.rotation_keys[&p]) < wide_rows));

    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    BootstrappingPresetRun::<BE>::setup(preset.clone(), None).check(backend);
    // Real slots take the S2C-first path of one EvalMod.
    BootstrappingPresetRun::<BE>::setup(preset.clone(), None)
        .with_real_slots()
        .check(backend);
    // EvalMod on the CI ring, with `u` as a product and folded, then all of it with
    // EvalRound+: CoeffsToSlots at 30 bits feeds EvalMod, the bypass at 50 sets the precision.
    // The keys of the first run, conversion keys included, go through their serialization.
    let eval_round_plus = preset.with_eval_round_plus(30, 50).unwrap();
    BootstrappingPresetRun::<BE>::setup(eval_round_plus.clone(), None).check(backend);
    let mut roundtrip = true;
    for preset in [&preset, &eval_round_plus] {
        for pre in [CIConversion::Product { log_delta: 46 }, CIConversion::Folded] {
            // With EvalRound+, `v` rounds `EvalMod(x) − K·x`, which holds the integer part.
            let post = if preset.plan().coeffs_to_slots_bypass().is_some() {
                48
            } else {
                38
            };
            let ci = EvalModRing::ConjugateInvariant {
                pre,
                post: CIConversion::Product { log_delta: post },
            };
            let preset = preset.with_eval_mod_ring(ci).unwrap();
            let ci = Module::<CIBackend<BE>>::new(preset.n() as u64);
            BootstrappingPresetRun::<BE>::setup_with_roundtrip(preset, Some(ci), std::mem::take(&mut roundtrip)).check(backend);
        }
    }
    // The transforms with the keys of power-of-two rotations, `u` folded on the CI ring.
    let powers_of_two = preset.with_power_of_two_keys().unwrap();
    BootstrappingPresetRun::<BE>::setup(powers_of_two.clone(), None).check(backend);
    let ci = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        post: CIConversion::Product { log_delta: 38 },
    };
    let preset = powers_of_two.with_eval_mod_ring(ci).unwrap();
    let ci = Module::<CIBackend<BE>>::new(preset.n() as u64);
    BootstrappingPresetRun::<BE>::setup(preset, Some(ci)).check(backend);
}

/// Runs the insecure `N = 2^11` C2S-first preset on `BE` and checks its key shapes
/// and output precision, then the same preset with EvalRound+. Small enough for a
/// default test run, so the keys a C2S-first plan derives, SlotsToCoeffs keys used
/// below EvalMod among them, run end to end. Backends register it when their radix
/// takes the preset's 52 bits.
pub fn small_c2s_preset_meets_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let preset = crate::presets::bootstrapping::insecure_n11_d35_k100_p17_c2s().unwrap();
    let keys = preset.keys_layout();
    let rows = |key: &dyn GGLWEInfos| key.dnum().as_usize();
    let wide_rows = keys.rotation_keys.values().map(|key| rows(key)).max().unwrap();
    // SlotsToCoeffs runs on what EvalMod leaves: its own keys get fewer rows than
    // those of the raised width.
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
    assert!(s2c_only.iter().all(|&p| rows(&keys.rotation_keys[&p]) < wide_rows));

    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    BootstrappingPresetRun::<BE>::setup(preset.clone(), None).check(backend);
    BootstrappingPresetRun::<BE>::setup(preset.with_eval_round_plus(30, 51).unwrap(), None).check(backend);
}

/// The run of the insecure `N = 2^11` S2C-first preset on the standard ring, for a backend that records its bootstrap.
pub fn small_s2c_preset_run_with_client<BE, BC>() -> BootstrappingPresetRun<BE, BC>
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static + CKKSCITwin,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE> + PresetRunModule<BE> + CKKSDFTOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::GLWEDiagonalProd<BE>,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    BootstrappingPresetRun::<BE, BC>::setup(crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap(), None)
}

/// The insecure `N = 2^11` S2C-first preset of [`small_s2c_preset_meets_precision`] on an
/// evaluated backend `BE` whose keys and ciphertexts come from the client backend `BC`: once on
/// the standard ring, and once with EvalMod on the conjugate-invariant ring, `u` folded and the
/// keys of power-of-two rotations. Small enough for a default test run, it moves every kind of
/// key and the ciphertexts both ways between two backends.
pub fn small_s2c_preset_meets_precision_with_client<BE, BC>()
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static + CKKSCITwin,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE> + PresetRunModule<BE> + CKKSDFTOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::GLWEDiagonalProd<BE>,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let preset = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    BootstrappingPresetRun::<BE, BC>::setup(preset.clone(), None).check(backend);
    let ci = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        post: CIConversion::Product { log_delta: 38 },
    };
    let folded = preset.with_power_of_two_keys().unwrap().with_eval_mod_ring(ci).unwrap();
    let ci = Module::<CIBackend<BE>>::new(folded.n() as u64);
    BootstrappingPresetRun::<BE, BC>::setup(folded, Some(ci)).check(backend);
}

/// Key generation from the plan alone ([`BootstrappingPlan::generate_keys`]) takes the keys the
/// matrices of the compiled context rotate with, Galois elements and first-factor rotations, and
/// the context's own key generation, from the same plan, the same keys, encapsulation and
/// conversion keys alike, on the
/// insecure `N = 2^11` S2C-first preset, with the keys of power-of-two rotations, and with EvalMod
/// on the conjugate-invariant ring, `u` folded into CoeffsToSlots.
pub fn bootstrapping_plan_keys_match_context<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k + CKKSCITwin,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let standard = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    let powers_of_two = standard.with_power_of_two_keys().unwrap();
    let folded = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        post: CIConversion::Product { log_delta: 38 },
    };
    let sources = || (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    for preset in [standard.clone(), powers_of_two.clone()] {
        let module = Module::<BE>::new(preset.n() as u64);
        let (sk_raw, _) = dense_secret(&module, &preset);
        let (plan, layout) = (preset.plan(), preset.keys_layout());
        let mut scratch = ScratchOwned::<BE>::alloc(bootstrap_setup_tmp_bytes(&module, &preset.bootstrap_layout(), plan, layout));
        let context =
            BootstrappingContextPrepared::<BE, f64>::compile(&module, None, preset.base2k().into(), plan, &mut scratch.borrow())
                .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_context = context
            .generate_keys(
                &module,
                None,
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_plan = plan
            .generate_keys(
                &module,
                None,
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_eq!(
            key_set_shape(&from_plan),
            key_set_shape(&from_context),
            "{}",
            preset_label(&preset)
        );
        // The plan's recipe against the rotations the compiled matrices evaluate.
        let shape = key_set_shape(&from_plan);
        assert_eq!(
            (shape.0, shape.1),
            compiled_key_rotations(&context, 2 * preset.n() as i64),
            "{}",
            preset_label(&preset)
        );
        // A layout without one of the rotation keys, or with the keyless identity, is refused.
        for bad in other_rotation_keys(layout) {
            let (mut xs, mut xa, mut xe) = sources();
            let Err(err) = context.generate_keys(&module, None, &sk_raw, &bad, &mut xs, &mut xe, &mut xa, &mut scratch.borrow())
            else {
                panic!("a layout of other rotation keys was accepted");
            };
            assert_refused_rotation_keys(&err, [xs, xa, xe], sources());
        }
    }
    for preset in [
        standard.with_eval_mod_ring(folded).unwrap(),
        powers_of_two.with_eval_mod_ring(folded).unwrap(),
    ] {
        let n = preset.n() as u64;
        let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
        let (sk_raw, _) = dense_secret(&module, &preset);
        let (plan, layout) = (preset.plan(), preset.keys_layout());
        let context = compile_context::<BE, f64>(&module, Some(&ci), preset.base2k().into(), plan);
        let mut scratch = ScratchOwned::<BE>::alloc(layout.generate_tmp_bytes(&module, Some(&ci)));
        let (mut xs, mut xa, mut xe) = sources();
        let from_context = context
            .generate_keys(
                &module,
                Some(&ci),
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_plan = plan
            .generate_keys(
                &module,
                Some(&ci),
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_eq!(
            key_set_shape(&from_plan),
            key_set_shape(&from_context),
            "{}",
            preset_label(&preset)
        );
        // The plan's recipe against the rotations the compiled matrices evaluate.
        let shape = key_set_shape(&from_plan);
        assert_eq!(
            (shape.0, shape.1),
            compiled_key_rotations(&context, 2 * preset.n() as i64),
            "{}",
            preset_label(&preset)
        );
        // A layout without one of the rotation keys, or with the keyless identity, is refused.
        for bad in other_rotation_keys(layout) {
            let (mut xs, mut xa, mut xe) = sources();
            let Err(err) = context.generate_keys(
                &module,
                Some(&ci),
                &sk_raw,
                &bad,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            ) else {
                panic!("a layout of other rotation keys was accepted");
            };
            assert_refused_rotation_keys(&err, [xs, xa, xe], sources());
        }
    }
}

/// `layout` without its last rotation key, and with the keyless identity added.
fn other_rotation_keys(layout: &BootstrappingKeysLayout) -> [BootstrappingKeysLayout; 2] {
    let mut missing = layout.clone();
    missing.rotation_keys.pop_last();
    let mut identity = layout.clone();
    identity
        .rotation_keys
        .insert(1, *layout.rotation_keys.values().next().unwrap());
    [missing, identity]
}

/// Asserts that key generation refused a layout of other rotation keys before drawing from its sources, given
/// as left by the refused call and freshly made.
fn assert_refused_rotation_keys(err: &anyhow::Error, [xs, xa, xe]: [Source; 3], fresh: (Source, Source, Source)) {
    assert!(
        err.to_string()
            .starts_with("bootstrapping key layout has rotation keys of Galois elements"),
        "{err}"
    );
    let (mut fresh_xs, mut fresh_xa, mut fresh_xe) = fresh;
    let (mut xs, mut xa, mut xe) = (xs, xa, xe);
    assert_eq!(
        [xs.new_seed(), xa.new_seed(), xe.new_seed()],
        [fresh_xs.new_seed(), fresh_xa.new_seed(), fresh_xe.new_seed()],
        "a refused generation drew from its sources"
    );
}

/// The Galois elements and first-factor rotations the matrices of a compiled context switch
/// with: the homomorphic DFTs, the bypass, the folded factors, and conjugation with EvalMod on
/// the standard ring.
fn compiled_key_rotations<BE: Backend, F>(
    context: &BootstrappingContextPrepared<BE, F>,
    cyclotomic_order: i64,
) -> (Vec<i64>, Vec<i64>) {
    let mut elements: std::collections::BTreeSet<i64> = context
        .coeffs_to_slots()
        .galois_elements(cyclotomic_order)
        .into_iter()
        .collect();
    if let Some(bypass) = context.coeffs_to_slots_bypass() {
        elements.extend(bypass.galois_elements(cyclotomic_order));
    }
    elements.extend(context.slots_to_coeffs().galois_elements(cyclotomic_order));
    if let Some([even, odd]) = context.eval_mod_ci().and_then(|ci| ci.pre().folded()) {
        elements.extend(even.galois_elements(cyclotomic_order));
        elements.extend(odd.galois_elements(cyclotomic_order));
    }
    if context.eval_mod_ring() == EvalModRing::Standard {
        elements.insert(-1);
    }
    let mut first_factor: Vec<i64> = context
        .coeffs_to_slots_first_factor()
        .map(|factor| factor.giant_steps.iter().map(|giant| giant.rot).collect())
        .unwrap_or_default();
    first_factor.sort_unstable();
    (elements.into_iter().collect(), first_factor)
}

/// The Galois elements of the rotation keys, the rotations of the first-factor keys,
/// and which encapsulation and conversion keys a key set holds.
fn key_set_shape<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord>(
    keys: &crate::layouts::BootstrappingKeySet<D, W>,
) -> (Vec<i64>, Vec<i64>, Option<bool>, bool) {
    let sorted = |mut v: Vec<i64>| {
        v.sort_unstable();
        v
    };
    (
        sorted(keys.rotation_keys.keys().copied().collect()),
        sorted(keys.first_factor_keys.keys().copied().collect()),
        keys.encapsulation_keys.as_ref().map(|(_, s2d)| s2d.is_some()),
        keys.conversion_keys.is_some(),
    )
}

/// The bootstrap with EvalMod on the conjugate-invariant ring refuses, before any
/// work: conversion keys narrower than its widths, a context compiled without
/// conversions, modules of the wrong degrees, and keys without the
/// dense-to-sparse key of an encapsulated recipe. On the insecure `N = 2^11`
/// preset with EvalMod on the CI ring.
pub fn ci_bootstrap_refuses_invalid_inputs<BE>()
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
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
        pre: CIConversion::Product { log_delta: 46 },
        post: CIConversion::Product { log_delta: 38 },
    };
    let preset = standard.with_eval_mod_ring(ring).unwrap();
    let n = preset.n() as u64;
    let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
    let sk_layout = preset.bootstrap_layout().glwe_layout;
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
    let context = compile_context::<BE, f64>(&module, Some(&ci), preset.base2k().into(), preset.plan());
    let compiled = {
        let bytes = bootstrap_setup_tmp_bytes(&module, &standard.bootstrap_layout(), standard.plan(), standard.keys_layout());
        with_scratch::<BE, _>(bytes, |s| {
            BootstrappingContextPrepared::<BE, f64>::compile(&module, None, preset.base2k().into(), standard.plan(), s)
        })
        .unwrap()
    };
    let bytes = bootstrap_setup_tmp_bytes(&module, &preset.bootstrap_layout(), preset.plan(), preset.keys_layout());
    let refused = with_scratch::<BE, _>(bytes, |s| {
        BootstrappingContextPrepared::<BE, f64>::compile(&module, None, preset.base2k().into(), preset.plan(), s)
    })
    .err()
    .expect("a CI plan without a CI module");
    assert!(matches!(
        refused.downcast_ref::<CKKSCompositionError>(),
        Some(CKKSCompositionError::InvalidOperand {
            op: "compile_unprepared",
            ..
        })
    ));
    let refused = with_scratch::<BE, _>(bytes, |s| {
        BootstrappingContextPrepared::<BE, f64>::compile(&module, Some(&ci), preset.base2k().into(), standard.plan(), s)
    })
    .err()
    .expect("a standard plan with a CI module");
    assert!(matches!(
        refused.downcast_ref::<CKKSCompositionError>(),
        Some(CKKSCompositionError::InvalidOperand {
            op: "compile_unprepared",
            ..
        })
    ));
    // Key generation leaves nothing of the CI secret, nor of the encryptions, in its arena: every
    // byte is wiped or was never carved.
    let mut scratch = poisoned_scratch::<BE>(preset.keys_layout().generate_tmp_bytes(&module, Some(&ci)));
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    context
        .generate_keys(
            &module,
            Some(&ci),
            &sk_raw,
            preset.keys_layout(),
            &mut source_xs,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        )
        .unwrap();
    assert!(
        BE::to_host_bytes(&scratch.data).iter().all(|&byte| byte == 0 || byte == 0xA5),
        "key generation left data in its scratch"
    );
    let keys_for = |layout: &BootstrappingKeysLayout| {
        prepare_keys_ci(
            &module,
            &ci,
            &generate_ci_keys(&module, &ci, &context, &sk_raw, layout),
            layout,
        )
    };
    let keys = keys_for(preset.keys_layout());
    // A module that does not match the ring of the plan, or of the key set, is refused.
    let key_set = generate_ci_keys(&module, &ci, &context, &sk_raw, preset.keys_layout());
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    for (op, refused) in [
        (
            "generate_keys",
            with_scratch::<BE, _>(0, |s| {
                preset
                    .plan()
                    .generate_keys(
                        &module,
                        None,
                        &sk_raw,
                        preset.keys_layout(),
                        &mut source_xs,
                        &mut source_xe,
                        &mut source_xa,
                        s,
                    )
                    .map(|_| ())
            }),
        ),
        (
            "prepare",
            with_scratch::<BE, _>(0, |s| key_set.prepare(&module, None, s).map(|_| ())),
        ),
        (
            "transfer_and_prepare",
            with_scratch::<BE, _>(0, |s| key_set.transfer_and_prepare(&module, None, s).map(|_| ())),
        ),
    ] {
        let refused = refused.unwrap_err();
        assert!(
            matches!(
                refused.downcast_ref::<CKKSCompositionError>(),
                Some(CKKSCompositionError::InvalidOperand { op: got, .. }) if *got == op
            ),
            "{op}: {refused}"
        );
    }
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
    let mut no_tensor_key = keys_for(preset.keys_layout());
    no_tensor_key.ci_tensor_key = None;

    let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&preset.input_layout());
    input.set_meta(preset.input_layout().meta());
    let mut output = module.ckks_ciphertext_alloc_from_glwe_infos(&preset.bootstrap_layout());
    // The bootstrap runs on exactly its query, and refuses an arena one alignment unit short.
    let bytes = module.ckks_bootstrap_tmp_bytes(
        &preset.bootstrap_layout(),
        &preset.input_layout(),
        &context,
        Some(&ci),
        preset.keys_layout(),
    );
    let mut bootstrap_on = |module: &Module<BE>,
                            context: &BootstrappingContextPrepared<BE, f64>,
                            keys: &crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
                            bytes: usize| {
        output.set_k(preset.bootstrap_k().into());
        let before = output.to_host_owned::<BE>();
        let result = with_scratch::<BE, _>(bytes, |s| {
            module.ckks_bootstrap(&mut output, &input, context, Some(&ci), keys, s)
        });
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
    let err = bootstrap_on(&module, &context, &keys, bytes - BE::SCRATCH_ALIGN).unwrap_err();
    assert!(matches!(err, crate::CKKSError::Internal(_)), "short arena: {err}");
    let mut bootstrap = |module: &Module<BE>,
                         context: &BootstrappingContextPrepared<BE, f64>,
                         keys: &crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE>| {
        bootstrap_on(module, context, keys, bytes)
    };
    bootstrap(&module, &context, &keys).unwrap();
    for (err, cause) in [
        (bootstrap(&module, &context, &narrow_keys[0]), "narrow trace key"),
        (bootstrap(&module, &context, &narrow_keys[1]), "narrow embed key"),
        (bootstrap(&module, &compiled, &keys), "context of compile"),
        (bootstrap(&Module::<BE>::new(n), &context, &keys), "module of degree N"),
        (bootstrap(&module, &context, &no_dense_to_sparse), "encapsulation"),
        (
            bootstrap(&module, &context, &no_tensor_key),
            "no tensor key of the plan's ring",
        ),
    ] {
        let err = err.unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand {
                    op: "ckks_bootstrap",
                    ..
                })
            ),
            "{cause}: {err}"
        );
    }
    let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut output, &input, &context, None, &keys, s)).unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::InvalidOperand { op: "ckks_bootstrap", reason })
                if reason.contains("a CI module goes with a plan of EvalMod on the CI ring")
        ),
        "no CI module: {err}"
    );
    // A working width narrower than the bits its stages consume, before any work.
    output.set_k(preset.base2k().into());
    let before = output.to_host_owned::<BE>();
    let err = with_scratch::<BE, _>(bytes, |s| {
        module.ckks_bootstrap(&mut output, &input, &context, Some(&ci), &keys, s)
    })
    .unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::InvalidOperand { op: "ckks_bootstrap", reason })
                if reason.contains("is narrower than the bits its stages consume")
        ),
        "narrow working width: {err}"
    );
    let after = output.to_host_owned::<BE>();
    assert!(
        (after.meta(), after.k(), after.data()) == (before.meta(), before.k(), before.data()),
        "a refusal wrote the output"
    );

    // The step alone refuses before its query, which reads these operands.
    output.set_k(preset.bootstrap_k().into());
    let boot = preset.bootstrap_layout();
    let run = context.run(preset.input_layout().meta());
    let mut wide = fixture_ciphertext(&module, &boot, 13);
    assert_eval_mod_refuses(
        &mut output,
        &mut wide,
        "a CI module goes with a plan of EvalMod on the CI ring, and only with one",
        |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, None, &keys, s),
    );
    let degrees = "takes a module of degree at least 2N and a CI module of degree N";
    let module_n = Module::<BE>::new(n);
    assert_eval_mod_refuses(&mut output, &mut wide, degrees, |res, a, s| {
        module_n.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci), &keys, s)
    });
    let ci_half = Module::<CIBackend<BE>>::new(n / 2);
    assert_eval_mod_refuses(&mut output, &mut wide, degrees, |res, a, s| {
        module.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci_half), &keys, s)
    });
    let mut half = boot;
    half.glwe_layout.n = (n as usize / 2).into();
    let (mut a_half, mut res_half) = (fixture_ciphertext(&module, &half, 17), fixture_ciphertext(&module, &half, 19));
    let key_degree = {
        let (trace, _) = keys.conversion_keys.as_ref().unwrap();
        let view = poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef::to_backend_ref(trace);
        crate::layouts::validation::validate_gadget_backend_view(
            "trace key",
            &trace.gglwe_layout(),
            &view,
            n as usize,
            a_half.base2k(),
            0,
        )
        .unwrap_err()
        .to_string()
    };
    assert_eval_mod_refuses(&mut res_half, &mut a_half, &key_degree, |res, a, s| {
        module.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci_half), &keys, s)
    });
    assert_eval_mod_refuses(
        &mut output,
        &mut wide,
        "the keys hold no tensor key of the plan's ring",
        |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci), &no_tensor_key, s),
    );
    let mut no_conversion_keys = keys_for(preset.keys_layout());
    no_conversion_keys.conversion_keys = None;
    assert_eval_mod_refuses(
        &mut output,
        &mut wide,
        "the keys hold no conversion keys of the plan's ring",
        |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci), &no_conversion_keys, s),
    );
    output.set_k((preset.bootstrap_k() - 1).into());
    assert_eval_mod_refuses(
        &mut output,
        &mut wide,
        "takes an input no wider than its output",
        |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, Some(&ci), &keys, s),
    );

    // The step on one arena of its query, in a composition that stamps nothing, gives the whole bootstrap, and refuses
    // conversion keys narrower than the widths it reaches.
    let fixture = fixture_ciphertext(&module, &preset.input_layout(), 5);
    let mut whole = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_bootstrap(&mut whole, &fixture, &context, Some(&ci), &keys, s)
    })
    .unwrap();
    let mut coeffs = module.ckks_ciphertext_alloc_from_infos(&fixture);
    let mut raised = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
    let mut stepped = fixture_ciphertext(&module, &boot, 23);
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_double_into(&mut coeffs, &fixture, s)?;
        module.ckks_slots_to_coeffs_assign(&mut coeffs, &run, &keys.rotation_keys, s)?;
        module.ckks_mod_raise(&mut raised, &mut coeffs, &run, &keys, s)?;
        module.ckks_coeffs_to_slots_assign(&mut raised, &run, &keys.rotation_keys, s)
    })
    .unwrap();
    // The embed key has one digit, so one limb of it goes rather than the digit.
    let narrow_embed_key = {
        let mut narrow = preset.keys_layout().clone();
        let (_, embed) = narrow.conversion_keys.as_mut().unwrap();
        embed.k_digit = (embed.k_digit.as_usize() - preset.base2k()).into();
        keys_for(&narrow)
    };
    for (narrow, cause) in [(&narrow_keys[0], "narrow trace key"), (&narrow_embed_key, "narrow embed key")] {
        let layout = narrow.keys_layout().unwrap();
        let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes(&stepped, &raised, &run, Some(&ci), layout);
        let before = snapshot::<BE, _>(&stepped);
        // The whole bootstrap refuses them before its first step, on an empty arena.
        let err = with_scratch::<BE, _>(0, |s| {
            module.ckks_bootstrap(&mut stepped, &fixture, &context, Some(&ci), narrow, s)
        })
        .unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand { op: "ckks_bootstrap", reason })
                    if reason.contains("a conversion key covers inputs of")
            ),
            "{cause}: {err}"
        );
        assert_eq!(snapshot::<BE, _>(&stepped), before, "a refused bootstrap wrote res");
        let err = with_scratch::<BE, _>(bytes, |s| {
            module.ckks_bootstrap_eval_mod(&mut stepped, &mut raised, &run, Some(&ci), narrow, s)
        })
        .unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand { op: "ckks_bootstrap_eval_mod", reason })
                    if reason.contains("a conversion key covers inputs of")
            ),
            "{cause}: {err}"
        );
        assert_eq!(snapshot::<BE, _>(&stepped), before, "a refused step wrote res");
    }
    let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes(&stepped, &raised, &run, Some(&ci), preset.keys_layout());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_bootstrap_eval_mod(&mut stepped, &mut raised, &run, Some(&ci), &keys, s)
    })
    .unwrap();
    assert_eq!(snapshot::<BE, _>(&stepped), snapshot::<BE, _>(&whole));
}

/// `Module<BE>` capabilities of the hand compositions of the bootstrap steps.
pub trait StepsRunModule<BE: TestContextBackend<Ring = Standard> + CKKSCITwin>:
    CKKSDFTMatrixPreparedFactory<BE>
    + GLWECIKeyEncryptSk<BE>
    + GLWECIKeyPreparedFactory<BE>
    + TestContextModule<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSBootstrappingOps<BE>
    + CKKSDFTOps<BE>
    + CKKSEvalModOps<BE>
    + CKKSDFTMatrixOps<BE, f64>
    + CKKSLinearTransformationOps<BE>
    + CKKSCopyOps<BE>
    + CKKSAddOps<BE>
    + GLWERotate<BE>
{
}

impl<BE: TestContextBackend<Ring = Standard> + CKKSCITwin, M> StepsRunModule<BE> for M where
    M: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSLinearTransformationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSAddOps<BE>
        + GLWERotate<BE>
{
}

/// A preset set up for hand compositions of the steps, each stage on an arena of exactly its query.
struct StepsRun<BE: CKKSCITwin> {
    preset: BootstrappingPreset,
    module: Module<BE>,
    sk_raw: BackendGLWESecret<BE>,
    sk: BackendGLWESecretPrepared<BE>,
    context: BootstrappingContextPrepared<BE, f64>,
    keys: BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    client_bytes: usize,
}

impl<BE> StepsRun<BE>
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn new(preset: BootstrappingPreset) -> Self {
        let module = Module::<BE>::new(preset.n() as u64);
        let keys_layout = preset.keys_layout();
        let (sk_raw, sk) = dense_secret(&module, &preset);
        let context = compile_context::<BE, f64>(&module, None, preset.base2k().into(), preset.plan());
        let (mut xs, mut xe, mut xa) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let key_set = with_scratch::<BE, _>(keys_layout.generate_tmp_bytes(&module, None), |s| {
            preset
                .plan()
                .generate_keys(&module, None, &sk_raw, keys_layout, &mut xs, &mut xe, &mut xa, s)
        })
        .unwrap();
        let keys = with_scratch::<BE, _>(keys_layout.prepare_tmp_bytes(&module, None), |s| {
            key_set.transfer_and_prepare(&module, None, s)
        })
        .unwrap();
        let client_bytes = bootstrap_setup_tmp_bytes(&module, &preset.bootstrap_layout(), preset.plan(), keys_layout);
        Self {
            preset,
            module,
            sk_raw,
            sk,
            context,
            keys,
            client_bytes,
        }
    }

    /// An encryption at the input layout of the preset of the slots `re + i·im`, labelled `slots`.
    fn encrypt(&self, re: &[f64], im: &[f64], slots: SlotsKind) -> CKKSCiphertextOwned<BE> {
        let (module, input) = (&self.module, self.preset.input_layout());
        let mut pt = module.ckks_plaintext_alloc_from_slots(self.preset.n() / 2, input.base2k(), input.k());
        pt.set_meta(GLWEPlaintextMeta { slots, ..input.meta() });
        let mut message = fixture_ciphertext(module, &input, 5);
        with_scratch::<BE, _>(self.client_bytes, |s| {
            module.ckks_encode_slots_host_into(&mut pt, re, im, s)?;
            let (mut xa, mut xe) = (Source::new([3; 32]), Source::new([4; 32]));
            module.ckks_encrypt_sk(&mut message, &pt, &self.sk, &mut xe, &mut xa, s)
        })
        .unwrap();
        message
    }

    /// The largest query of `dft` over the rotation keys, on ciphertexts of the layout of `ct`.
    fn dft_bytes<Dir, Fmt, D>(&self, ct: &CKKSCiphertextOwned<BE>, dft: &D) -> usize
    where
        D: CKKSDFTFactors<BE, Dir, Fmt>,
    {
        self.preset
            .keys_layout()
            .rotation_key_layouts()
            .iter()
            .map(|atk| self.module.ckks_dft_tmp_bytes(ct, ct, dft, atk))
            .max()
            .unwrap_or(0)
    }

    /// `message` doubled and through SlotsToCoeffs: `2·x` through the split decoding matrix.
    fn to_coeffs(&self, message: &CKKSCiphertextOwned<BE>) -> CKKSCiphertextOwned<BE> {
        let module = &self.module;
        let mut coeffs = module.ckks_ciphertext_alloc_from_infos(message);
        let bytes = self
            .dft_bytes::<Decode, Split, _>(message, &self.context)
            .max(module.ckks_double_tmp_bytes(message.size()));
        with_scratch::<BE, _>(bytes, |s| {
            module.ckks_double_into(&mut coeffs, message, s)?;
            module.ckks_slots_to_coeffs_assign(&mut coeffs, &self.context, &self.keys.rotation_keys, s)
        })
        .unwrap();
        coeffs
    }

    /// The real-slot stages above SlotsToCoeffs by hand: the raise of `coeffs`, CoeffsToSlots, with `mask` in place of
    /// its last factor when there is one, the label the transforms lose, the EvalMod of a bootstrap and the metadata of
    /// the message.
    fn raise_to_eval_mod(
        &self,
        coeffs: &CKKSCiphertextOwned<BE>,
        mask: Option<&CoeffsMaskPrepared<BE>>,
        meta: GLWEPlaintextMeta,
    ) -> CKKSCiphertextOwned<BE> {
        let (module, context, keys) = (&self.module, &self.context, &self.keys);
        let (boot, keys_layout) = (self.preset.bootstrap_layout(), self.preset.keys_layout());
        let mut a = module.ckks_ciphertext_alloc_from_infos(coeffs);
        with_scratch::<BE, _>(module.ckks_copy_tmp_bytes(&a, coeffs), |s| {
            module.ckks_copy(&mut a, coeffs, s)
        })
        .unwrap();
        let mut z = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
        let mut res = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
        with_scratch::<BE, _>(module.ckks_mod_raise_tmp_bytes(&z, &a, context, keys_layout), |s| {
            module.ckks_mod_raise(&mut z, &mut a, context, keys, s)
        })
        .unwrap();
        match mask {
            Some(mask) => {
                let masked = context.masked(mask).unwrap();
                let bytes = self.dft_bytes(&z, &masked);
                with_scratch::<BE, _>(bytes, |s| {
                    module.ckks_coeffs_to_slots_assign(&mut z, &masked, &keys.rotation_keys, s)
                })
            }
            None => {
                let bytes = self.dft_bytes::<Encode, Split, _>(&z, context);
                with_scratch::<BE, _>(bytes, |s| {
                    module.ckks_coeffs_to_slots_assign(&mut z, context, &keys.rotation_keys, s)
                })
            }
        }
        .unwrap();
        z.set_slots(SlotsKind::Real);
        let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes(&res, &z, context, None, keys_layout);
        with_scratch::<BE, _>(bytes, |s| {
            module.ckks_bootstrap_eval_mod(&mut res, &z, context, None, keys, s)
        })
        .unwrap();
        res.set_meta(meta);
        res
    }

    /// `ct` decrypts to the slots `re + i·im` within 14 bits, at the scale of the input.
    fn assert_decrypts(&self, label: &str, ct: &CKKSCiphertextOwned<BE>, re: &[f64], im: &[f64]) {
        let (module, half) = (&self.module, self.preset.n() / 2);
        let (base2k, log_delta) = (
            self.preset.bootstrap_layout().base2k(),
            self.preset.input_layout().log_delta(),
        );
        let mut got = module.ckks_plaintext_alloc_from_slots(half, base2k, (log_delta + 60).into());
        got.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(ct.log_delta()),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        let (mut got_re, mut got_im) = (vec![0.0; half], vec![0.0; half]);
        assert_canonical_at_k::<BE>(label, ct);
        with_scratch::<BE, _>(self.client_bytes, |s| {
            module.ckks_decrypt(&mut got, ct, &self.sk, s)?;
            module.ckks_decode_slots_host_into(&got, &mut got_re, &mut got_im, s)
        })
        .unwrap();
        let (got_re, got_im) = (
            precision_stats(&got_re, re, log_delta),
            precision_stats(&got_im, im, log_delta),
        );
        // Measured at 22.8 bits at worst on the toy presets: a margin of about 8 bits.
        assert!(
            got_re.min_log2_prec.min(got_im.min_log2_prec) >= 14.0,
            "{label}: {:.2} and {:.2} bits",
            got_re.min_log2_prec,
            got_im.min_log2_prec
        );
    }
}

/// The steps by hand, on real slots, give the digits, layout and metadata of [`CKKSBootstrappingOps::ckks_bootstrap`],
/// each on an arena of exactly its query, and [`BootstrappingContextPrepared::masked`] zeroes the coefficients a mask
/// leaves out, read from either half of the raise's input.
pub fn test_ckks_coeffs_mask<BE>()
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let run = StepsRun::<BE>::new(crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap());
    let (module, context, keys) = (&run.module, &run.context, &run.keys);
    let n = run.preset.n();
    let half = n / 2;
    let boot = run.preset.bootstrap_layout();
    let keys_layout = run.preset.keys_layout();

    let (want, zeros) = (test_vector_1::<f64>(half).0, vec![0.0; half]);
    let message = run.encrypt(&want, &zeros, SlotsKind::Real);
    let mut whole = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
    let bytes = module.ckks_bootstrap_tmp_bytes(&boot, &message, context, None, keys_layout);
    with_scratch::<BE, _>(bytes, |s| module.ckks_bootstrap(&mut whole, &message, context, None, keys, s)).unwrap();

    let coeffs = run.to_coeffs(&message);
    let staged = run.raise_to_eval_mod(&coeffs, None, message.meta());
    assert_eq!(snapshot::<BE, _>(&staged), snapshot::<BE, _>(&whole));

    // Slot `j` holds coefficient `bitrev(j)` of SlotsToCoeffs.
    let keep: Vec<bool> = (0..n).map(|j| j < half && j % 5 != 2).collect();
    let bytes = context.coeffs_mask_tmp_bytes(module).unwrap();
    let mask = with_scratch::<BE, _>(bytes, |s| context.coeffs_mask(module, &keep, s)).unwrap();
    let bitrev = |j: usize| j.reverse_bits() >> (usize::BITS - half.trailing_zeros());
    let masked: Vec<f64> = (0..half).map(|j| if keep[bitrev(j)] { want[j] } else { 0.0 }).collect();
    let got = run.raise_to_eval_mod(&coeffs, Some(&mask), message.meta());
    run.assert_decrypts("masked", &got, &masked, &zeros);

    // The same coefficients one half up, read through the mask.
    let up: Vec<bool> = (0..n).map(|j| keep[(j + half) % n]).collect();
    let mask = with_scratch::<BE, _>(bytes, |s| context.coeffs_mask(module, &up, s)).unwrap();
    let mut rotated = module.ckks_ciphertext_alloc_from_infos(&coeffs);
    let rotate_bytes = module
        .ckks_copy_tmp_bytes(&coeffs, &coeffs)
        .max(module.glwe_rotate_tmp_bytes());
    with_scratch::<BE, _>(rotate_bytes, |s| {
        module.ckks_copy(&mut rotated, &coeffs, s)?;
        module.glwe_rotate_assign(half as i64, &mut rotated, s);
        Ok::<_, anyhow::Error>(())
    })
    .unwrap();
    let got = run.raise_to_eval_mod(&rotated, Some(&mask), message.meta());
    run.assert_decrypts("masked, one half up", &got, &masked, &zeros);
}

/// The toy preset of [`insecure_n11_d35_k180_p20_s2c`](crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c) on
/// plaintexts of gap 2.
fn toy_subring_preset() -> BootstrappingPreset {
    crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c()
        .unwrap()
        .with_log_gap(1, &[(3, 1), (3, 32), (3, 4)], &[(2, 2), (3, 16), (4, 128)])
        .unwrap()
}

/// The last SlotsToCoeffs factor with one kernel per coset finishes SlotsToCoeffs and multiplies each coset of the
/// coefficients by its kernel, in the level of that factor, on the full ring and in a subring.
pub fn test_merged_s2c_factor<BE>()
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    merged_factor_on::<BE>(crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap());
    merged_factor_on::<BE>(toy_subring_preset());
}

fn merged_factor_on<BE>(preset: BootstrappingPreset)
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let run = StepsRun::<BE>::new(preset);
    let (module, context, keys) = (&run.module, &run.context, &run.keys);
    let n = run.preset.n();
    let half = n / 2;
    let gap = 1usize << run.preset.plan().log_gap();
    let slots = half / gap;

    let values = test_vector_1::<f64>(slots).0;
    let (want, zeros): (Vec<f64>, _) = ((0..half).map(|j| values[j % slots]).collect(), vec![0.0; half]);
    let x = run.encrypt(&want, &zeros, SlotsKind::Real);

    // Coefficient `bitrev(j)` is slot `j`, and the coset of a coefficient picks its kernel.
    let cosets = context.merged_s2c_cosets();
    let mut source = Source::new([11; 32]);
    let kernels: Vec<Vec<f64>> = (0..cosets)
        .map(|_| {
            let mut kernel = vec![0.0; n];
            for _ in 0..4 {
                let e = source.next_u64n((n / gap) as u64, (n / gap) as u64 - 1) as usize;
                kernel[gap * e] = source.next_f64(-0.25, 0.25);
            }
            kernel
        })
        .collect();
    let bitrev = |j: usize| j.reverse_bits() >> (usize::BITS - slots.trailing_zeros());
    let mut product = vec![0.0; n];
    for (j, &value) in values.iter().enumerate() {
        let (k, idx) = (bitrev(j), gap * bitrev(j));
        for (e, &w) in kernels[k % cosets].iter().enumerate().filter(|&(_, &w)| w != 0.0) {
            let (d, sign) = if idx + e < n { (idx + e, 1.0) } else { (idx + e - n, -1.0) };
            product[d] += sign * w * value;
        }
    }

    let log_delta_w = run.preset.plan().slots_to_coeffs.coeffs_meta.log_delta();
    let width = log_delta_w + run.preset.base2k();
    let bytes = context.merged_s2c_factor_tmp_bytes(module, width).unwrap();
    let merged = with_scratch::<BE, _>(bytes, |s| context.merged_s2c_factor(module, &kernels, log_delta_w, s)).unwrap();

    // SlotsToCoeffs up to its last factor, `2·x` through the split decoding matrix, then the merged factor.
    let plan = merged.factor().index();
    let head = context.slots_to_coeffs().head();
    let bytes = run
        .dft_bytes::<Decode, Split, _>(&x, context)
        .max(module.ckks_double_tmp_bytes(x.size()))
        + module.glwe_linear_transformation_baby_steps_bytes_of(&plan, &x);
    let mut running = module.ckks_ciphertext_alloc_from_infos(&x);
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_double_into(&mut running, &x, s)?;
        module.ckks_slots_to_coeffs_assign(&mut running, &head, &keys.rotation_keys, s)?;
        let (mut babies, mut s) = s
            .borrow()
            .take_glwe_linear_transformation_baby_steps_scratch(module, &plan, &running);
        module.ckks_linear_transformation_baby_steps_prepare(&mut babies, &running, &keys.rotation_keys, &mut s)?;
        module.ckks_linear_transformation_eval_assign(&mut running, &babies, merged.factor(), &keys.rotation_keys, &mut s)
    })
    .unwrap();

    let mut got = module.ckks_plaintext_alloc_from_infos(&running);
    got.set_meta(running.meta());
    let mut coeffs = vec![0.0; n];
    assert_canonical_at_k::<BE>("merged factor", &running);
    with_scratch::<BE, _>(run.client_bytes, |s| {
        module.ckks_decrypt(&mut got, &running, &run.sk, s)?;
        module.ckks_decode_coeffs_host_into(&got, &mut coeffs, s)
    })
    .unwrap();
    let err = coeffs.iter().zip(&product).fold(0.0f64, |e, (g, w)| e.max((g - w).abs()));
    assert!(err < (-10.0f64).exp2(), "merged factor against the coset product: {err:e}");
}

/// The steps of a context of gap 2: slots that repeat every `N/4`, through SlotsToCoeffs and the raise, come back through
/// a mask, and what a layer leaves off the subring goes with the raise; the whole bootstrap of the context on real and
/// on complex slots. Keys of a subring need a bound at its degree.
pub fn test_ckks_coeffs_mask_subring<BE>()
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let run = StepsRun::<BE>::new(toy_subring_preset());
    let (module, context, keys) = (&run.module, &run.context, &run.keys);
    let n = run.preset.n();
    let (half, slots) = (n / 2, n / 4);
    let boot = run.preset.bootstrap_layout();
    let keys_layout = run.preset.keys_layout();

    let values = test_vector_1::<f64>(slots).0;
    let (want, zeros): (Vec<f64>, _) = ((0..half).map(|j| values[j % slots]).collect(), vec![0.0; half]);
    let message = run.encrypt(&want, &zeros, SlotsKind::Real);

    // The same coefficients one up land on the odd indexes.
    let mut coeffs = run.to_coeffs(&message);
    let mut off_subring = module.ckks_ciphertext_alloc_from_infos(&coeffs);
    let garbage_bytes = module
        .ckks_copy_tmp_bytes(&coeffs, &coeffs)
        .max(module.glwe_rotate_tmp_bytes())
        .max(module.ckks_add_tmp_bytes(coeffs.size()));
    with_scratch::<BE, _>(garbage_bytes, |s| {
        module.ckks_copy(&mut off_subring, &coeffs, s)?;
        module.glwe_rotate_assign(1, &mut off_subring, s);
        module.ckks_add_assign(&mut coeffs, &off_subring, s)
    })
    .unwrap();

    // Slot `s` holds coefficient `2·bitrev(s)` of SlotsToCoeffs, and the slots repeat every `N/4`.
    let keep: Vec<bool> = (0..n).map(|j| j < half && j.is_multiple_of(2) && j % 10 != 4).collect();
    let bytes = context.coeffs_mask_tmp_bytes(module).unwrap();
    let mask = with_scratch::<BE, _>(bytes, |s| context.coeffs_mask(module, &keep, s)).unwrap();
    let bitrev = |s: usize| s.reverse_bits() >> (usize::BITS - slots.trailing_zeros());
    let masked: Vec<f64> = (0..half)
        .map(|j| if keep[2 * bitrev(j % slots)] { want[j] } else { 0.0 })
        .collect();
    assert!(masked.iter().zip(&want).any(|(m, w)| m != w), "the mask drops a coefficient");
    let got = run.raise_to_eval_mod(&coeffs, Some(&mask), message.meta());
    run.assert_decrypts("masked in a subring", &got, &masked, &zeros);

    // The same coefficients one half up, read through the mask.
    let up: Vec<bool> = (0..n).map(|j| keep[(j + half) % n]).collect();
    let mask = with_scratch::<BE, _>(bytes, |s| context.coeffs_mask(module, &up, s)).unwrap();
    let mut rotated = module.ckks_ciphertext_alloc_from_infos(&coeffs);
    with_scratch::<BE, _>(garbage_bytes, |s| {
        module.ckks_copy(&mut rotated, &coeffs, s)?;
        module.glwe_rotate_assign(half as i64, &mut rotated, s);
        Ok::<_, anyhow::Error>(())
    })
    .unwrap();
    let got = run.raise_to_eval_mod(&rotated, Some(&mask), message.meta());
    run.assert_decrypts("masked in a subring, one half up", &got, &masked, &zeros);

    // Key generation leaves nothing of the secret of the subring in its arena: every byte is wiped or was never carved.
    let mut scratch = poisoned_scratch::<BE>(keys_layout.generate_tmp_bytes(module, None));
    let (mut xs, mut xe, mut xa) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    run.preset
        .plan()
        .generate_keys(
            module,
            None,
            &run.sk_raw,
            keys_layout,
            &mut xs,
            &mut xe,
            &mut xa,
            &mut scratch.arena(),
        )
        .unwrap();
    assert!(
        BE::to_host_bytes(&scratch.data).iter().all(|&byte| byte == 0 || byte == 0xA5),
        "key generation left data in its scratch"
    );

    let imag = test_vector_1::<f64>(slots).1;
    let imag: Vec<f64> = (0..half).map(|j| imag[j % slots]).collect();
    for (kind, want_im) in [(SlotsKind::Real, &zeros), (SlotsKind::Complex, &imag)] {
        let message = run.encrypt(&want, want_im, kind);
        let mut whole = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
        let bytes = module.ckks_bootstrap_tmp_bytes(&boot, &message, context, None, keys_layout);
        with_scratch::<BE, _>(bytes, |s| module.ckks_bootstrap(&mut whole, &message, context, None, keys, s)).unwrap();
        run.assert_decrypts(&format!("bootstrap in a subring, {kind:?} slots"), &whole, &want, want_im);
    }
}

/// Pins a setup refusal: [`CKKSCompositionError::InvalidOperand`] of `op` for `reason`.
fn assert_refused<T>(label: &str, result: anyhow::Result<T>, op: &str, reason: &str) {
    let err = result.err().unwrap_or_else(|| panic!("{label}: not refused"));
    assert!(
        matches!(err.downcast_ref::<CKKSCompositionError>(),
            Some(CKKSCompositionError::InvalidOperand { op: got, reason: why }) if *got == op && why.contains(reason)),
        "{label}: unexpected error: {err}"
    );
}

/// Pins a refusal of a gap the plan does not fit: [`CKKSCompositionError::InvalidPlan`] for `reason`.
fn assert_refused_gap<T>(label: &str, result: anyhow::Result<T>, reason: &str) {
    let err = result.err().unwrap_or_else(|| panic!("{label}: not refused"));
    assert!(
        matches!(err.downcast_ref::<CKKSCompositionError>(),
            Some(CKKSCompositionError::InvalidPlan { plan: "BootstrappingPlan", reason: why }) if why.contains(reason)),
        "{label}: unexpected error: {err}"
    );
}

/// A raise of the plan `plan` carrying the first factor of `prepared`, whatever `plan` says of it.
struct RaiseWithFactor<'a, BE: Backend> {
    plan: ModRaisePlan,
    prepared: &'a ModRaisePrepared<BE>,
}

impl<BE: Backend> CKKSModRaiseInfos<BE> for RaiseWithFactor<'_, BE> {
    fn mod_raise_plan(&self) -> &ModRaisePlan {
        &self.plan
    }

    fn first_factor(&self) -> Option<&poulpy_core::layouts::LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        self.prepared.first_factor()
    }
}

/// The mask, its view, the merged factor and the gap refuse with a typed error, before the arena check on an arena of no
/// byte: a context of another pipeline or ring, under EvalRound+ or a LUT, or that evaluates no factor, a mask or kernels
/// of another shape, or off the subring, a mask of another factor or plan, a gap off the slots of the transforms, a sparse
/// plan without its gap, and a raise of a gap without what it reads. The arena check comes last.
pub fn test_ckks_coeffs_mask_rejects_invalid_inputs<BE>()
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin + CKKSModUpFirstFactorImpl + HostStaged,
    CIBackend<BE>: TestContextBackend,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    Module<BE>: StepsRunModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let run = StepsRun::<BE>::new(crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap());
    let (module, preset, context) = (&run.module, &run.preset, &run.context);
    let n = preset.n();
    let half = n / 2;
    let base2k = Base2K::from(preset.base2k());
    let compile = |plan: &BootstrappingPlan| compile_context::<BE, f64>(module, None, base2k, plan);

    let c2s_first = crate::presets::bootstrapping::insecure_n11_d35_k100_p17_c2s().unwrap();
    let c2s_context = compile(c2s_first.plan());
    let subring_run = StepsRun::<BE>::new(toy_subring_preset());
    let subring_context = &subring_run.context;
    // A CoeffsToSlots of one factor, left to the raise.
    let plan = preset.plan();
    let raised_only = BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        plan.techniques().clone(),
        DFTPlan::new(
            DFTType::Encode,
            vec![(preset.log_n() - 1, 1)],
            DFTOutputFormat::SplitRealAndImag,
            plan.coeffs_to_slots().coeffs_meta(),
        )
        .unwrap()
        .with_first_factor_on_raise(),
        *plan.eval_mod(),
        plan.slots_to_coeffs().clone(),
    )
    .unwrap();
    let raised_only = compile(&raised_only);
    let ci_ring = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Product { log_delta: 46 },
        post: CIConversion::Product { log_delta: 38 },
    };
    let ci_preset = preset.with_eval_mod_ring(ci_ring).unwrap();
    let (module_2n, ci) = (Module::<BE>::new(2 * n as u64), Module::<CIBackend<BE>>::new(n as u64));
    let ci_context = compile_context::<BE, f64>(&module_2n, Some(&ci), base2k, ci_preset.plan());
    let round_context = compile(preset.with_eval_round_plus(30, 50).unwrap().plan());
    let host = Module::<HostBytesBackend>::new(n as u64);
    let lut = EncodedLut::binary(
        &host,
        3.0f64,
        1.0,
        8,
        4,
        1,
        base2k,
        CoeffsMeta::from_delta_budget(16, preset.base2k()),
        SplitStrategy::MinDepth,
    )
    .unwrap();
    let functional_context = compile(&plan.clone().with_functional_bootstrap(&lut).unwrap());

    const MASK: &str = "coeffs_mask";
    let keep: Vec<bool> = (0..n).map(|j| j < half && j.is_multiple_of(2)).collect();
    let mask_of = |context: &BootstrappingContextPrepared<BE, f64>, module: &Module<BE>, keep: &[bool]| {
        with_scratch::<BE, _>(0, |s| context.coeffs_mask(module, keep, s))
    };
    let identity = "takes an identity S2C-first context";
    assert_refused("another pipeline", mask_of(&c2s_context, module, &keep), MASK, identity);
    assert_refused("the CI ring", mask_of(&ci_context, &module_2n, &keep), MASK, identity);
    assert_refused("EvalRound+", mask_of(&round_context, module, &keep), MASK, identity);
    assert_refused("a LUT", mask_of(&functional_context, module, &keep), MASK, identity);
    assert_refused("no factor", mask_of(&raised_only, module, &keep), MASK, "evaluates no factor");
    assert_refused(
        "another length",
        mask_of(context, module, &keep[1..]),
        MASK,
        "one entry per coefficient",
    );
    assert_refused(
        "both halves",
        mask_of(context, module, &vec![true; n]),
        MASK,
        "one of the two coefficients",
    );
    let odd: Vec<bool> = (0..n).map(|j| j < half && !j.is_multiple_of(2)).collect();
    assert_refused("off the subring", mask_of(subring_context, module, &odd), MASK, "subring");
    let bytes = context.coeffs_mask_tmp_bytes(module).unwrap();
    let err = with_scratch::<BE, _>(bytes - 1, |s| context.coeffs_mask(module, &keep, s))
        .err()
        .unwrap();
    assert!(err.to_string().contains("needs a scratch arena"), "{err}");
    let err = raised_only.coeffs_mask_tmp_bytes(module).err().unwrap();
    assert!(err.to_string().contains("evaluates no factor"), "{err}");

    const MERGED: &str = "merged_s2c_factor";
    let factor_of = |context: &BootstrappingContextPrepared<BE, f64>, kernels: &[Vec<f64>]| {
        with_scratch::<BE, _>(0, |s| context.merged_s2c_factor(module, kernels, 20, s))
    };
    let cosets = context.merged_s2c_cosets();
    assert_refused(
        "another pipeline",
        factor_of(&c2s_context, &[]),
        MERGED,
        "takes an S2C-first context",
    );
    assert_refused("no kernel", factor_of(context, &[]), MERGED, "kernel count");
    assert_refused(
        "a count off the cosets",
        factor_of(context, &vec![vec![0.0; n]; cosets + 1]),
        MERGED,
        "kernel count",
    );
    assert_refused(
        "another degree",
        factor_of(context, &vec![vec![0.0; half]; cosets]),
        MERGED,
        "degree of the bootstrap",
    );
    let mut off_subring = vec![0.0; n];
    off_subring[1] = 1.0;
    assert_refused(
        "off the subring",
        factor_of(subring_context, &[off_subring]),
        MERGED,
        "subring",
    );
    let bytes = context.merged_s2c_factor_tmp_bytes(module, 20).unwrap();
    let kernels = vec![vec![0.0; n]; cosets];
    let err = with_scratch::<BE, _>(bytes - 1, |s| context.merged_s2c_factor(module, &kernels, 20, s))
        .err()
        .unwrap();
    assert!(err.to_string().contains("needs a scratch arena"), "{err}");

    // A kernel per coset, on the degree of the bootstrap of a module of `2N`, keeps the rotations of the factor.
    let bytes = ci_context
        .merged_s2c_factor_tmp_bytes(&module_2n, 20 + preset.base2k())
        .unwrap();
    let merged = with_scratch::<BE, _>(bytes, |s| ci_context.merged_s2c_factor(&module_2n, &kernels, 20, s)).unwrap();
    let last = ci_context.slots_to_coeffs().factor_operands().last().unwrap();
    assert!(merged.factor().index() == last.index(), "a merged factor keeps its schedule");

    let bytes = subring_context.coeffs_mask_tmp_bytes(module).unwrap();
    let mask = with_scratch::<BE, _>(bytes, |s| subring_context.coeffs_mask(module, &keep, s)).unwrap();
    assert_invalid_operand(
        "a mask of another factor",
        context.masked(&mask).err().as_ref().unwrap(),
        "masked",
        "not of the last CoeffsToSlots factor",
    );
    assert_invalid_operand(
        "a context with no factor",
        raised_only.masked(&mask).err().as_ref().unwrap(),
        "masked",
        "evaluates no factor",
    );
    let own_mask = {
        let bytes = context.coeffs_mask_tmp_bytes(module).unwrap();
        with_scratch::<BE, _>(bytes, |s| context.coeffs_mask(module, &keep, s)).unwrap()
    };
    assert!(context.masked(&own_mask).is_ok());
    for (label, other) in [("EvalRound+", &round_context), ("a LUT", &functional_context)] {
        assert_invalid_operand(label, other.masked(&own_mask).err().as_ref().unwrap(), "masked", identity);
    }
    // One schedule, but the scaling of another interval, or another scale: the view reads the mask's plan.
    let scales = PresetScales {
        c2s_log_delta: preset.scales().c2s_log_delta - 2,
        ..preset.scales()
    };
    for other in [
        preset.with_eval_mod_polynomial(8, 30, 3).unwrap(),
        preset.with_scales(scales).unwrap(),
    ] {
        let other_context = compile(other.plan());
        let bytes = other_context.coeffs_mask_tmp_bytes(module).unwrap();
        let other_mask = with_scratch::<BE, _>(bytes, |s| other_context.coeffs_mask(module, &keep, s)).unwrap();
        let (last, other_last) = (
            context.coeffs_to_slots().factor_operands().last().unwrap(),
            other_context.coeffs_to_slots().factor_operands().last().unwrap(),
        );
        assert!(last.index() == other_last.index(), "the case needs one schedule");
        assert_invalid_operand(
            "a mask of another plan",
            context.masked(&other_mask).err().as_ref().unwrap(),
            "masked",
            "not of the last CoeffsToSlots factor",
        );
    }

    assert_refused_gap(
        "a plan without the first factor on the raise",
        preset.plan().clone().with_log_gap(1),
        "a gap takes an S2C-first plan",
    );
    let wider_gap = subring_run.preset.plan().clone().with_log_gap(2).unwrap();
    assert_refused_gap(
        "a compile past the slots",
        with_scratch::<BE, _>(0, |s| {
            BootstrappingContextPrepared::<BE, f64>::compile_unprepared(module, None, base2k, &wider_gap, s)
        }),
        "do not fit the module",
    );
    // A module may exceed the degree of the bootstrap, with a gap as without one.
    compile_context::<BE, f64>(&module_2n, None, base2k, subring_run.preset.plan());
    let (mut xs, mut xe, mut xa) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    assert_refused_gap(
        "keys off the slots",
        with_scratch::<BE, _>(0, |s| {
            wider_gap.generate_keys(module, None, &run.sk_raw, preset.keys_layout(), &mut xs, &mut xe, &mut xa, s)
        }),
        "must fill the slots of the subring",
    );

    // A plan on fewer slots than the ring, without its gap: keys and bootstraps refuse what the compile lets through.
    let sparse = subring_run.preset.plan().clone().with_log_gap(0).unwrap();
    assert_refused_gap(
        "keys of a sparse plan",
        with_scratch::<BE, _>(0, |s| {
            sparse.generate_keys(module, None, &run.sk_raw, preset.keys_layout(), &mut xs, &mut xe, &mut xa, s)
        }),
        "must fill the slots of the subring",
    );
    let sparse_context = compile(&sparse);
    let (mut out, input) = (
        fixture_ciphertext(module, &subring_run.preset.bootstrap_layout(), 8),
        fixture_ciphertext(module, &subring_run.preset.input_layout(), 9),
    );
    let before = snapshot::<BE, _>(&out);
    let degree = "must fill the slots of the input's degree";
    let err = with_scratch::<BE, _>(0, |s| {
        module.ckks_bootstrap(&mut out, &input, &sparse_context, None, &subring_run.keys, s)
    })
    .unwrap_err();
    assert_invalid_operand("a bootstrap of a sparse plan", &err, "ckks_bootstrap", degree);
    let functional_sparse = compile(&sparse.clone().with_functional_bootstrap(&lut).unwrap());
    let luts = [lut.transfer_to(module)];
    let mut outs = vec![out];
    let err = with_scratch::<BE, _>(0, |s| {
        module.ckks_functional_bootstrap(&mut outs, &input, &functional_sparse, &luts, &subring_run.keys, s)
    })
    .unwrap_err();
    assert_invalid_operand(
        "a functional bootstrap of a sparse plan",
        &err,
        "ckks_functional_bootstrap",
        degree,
    );
    assert_eq!(snapshot::<BE, _>(&outs[0]), before, "a refused bootstrap wrote its output");

    // A raise whose plan claims a gap without the rest of what a gap reads.
    let (mut res, mut a) = (
        fixture_ciphertext(module, &subring_run.preset.bootstrap_layout(), 6),
        fixture_ciphertext(module, &subring_run.preset.input_layout(), 7),
    );
    let (keys, raise) = (&subring_run.keys, subring_context.mod_raise());
    let reason = "a gap takes an S2C-first raise";
    let no_factor = ModRaisePlan {
        first_factor_on_raise: false,
        ..raise.mod_raise_plan().clone()
    };
    assert_mod_raise_refuses(&mut res, &mut a, reason, |res, a, s| {
        module.ckks_mod_raise(res, a, &no_factor, keys, s)
    });
    let lifted = RaiseWithFactor {
        plan: ModRaisePlan {
            lift: Some(*preset.plan().eval_mod()),
            ..raise.mod_raise_plan().clone()
        },
        prepared: raise,
    };
    assert_mod_raise_refuses(&mut res, &mut a, reason, |res, a, s| {
        module.ckks_mod_raise(res, a, &lifted, keys, s)
    });
    let mut subring_run = subring_run;
    let raise_plan = ModRaisePlan {
        sparse_secret_hamming_weight: None,
        ..subring_run.context.mod_raise().mod_raise_plan().clone()
    };
    subring_run.keys.encapsulation_keys = None;
    let (keys, context) = (&subring_run.keys, &subring_run.context);
    let plain = RaiseWithFactor {
        plan: raise_plan,
        prepared: context.mod_raise(),
    };
    assert_mod_raise_refuses(&mut res, &mut a, reason, |res, a, s| {
        module.ckks_mod_raise(res, a, &plain, keys, s)
    });
    let factor = context.coeffs_to_slots_first_factor().unwrap();
    let err = with_scratch::<BE, _>(0, |s| {
        BE::ckks_mod_up_first_factor(module, &mut res, &mut a, 0, factor, 1, keys, s)
    })
    .unwrap_err();
    assert_invalid_operand("the hook", &err, "ckks_mod_raise", "a gap needs the sparse secret");
}

/// [`BootstrappingContextPrepared::compile`] of `plan`, each stage on an arena of exactly its query.
pub(crate) fn compile_context<BE, F>(
    module: &Module<BE>,
    ci: Option<&Module<CIBackend<BE>>>,
    base2k: Base2K,
    plan: &BootstrappingPlan,
) -> BootstrappingContextPrepared<BE, F>
where
    BE: CKKSCITwin,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + crate::api::CKKSRightOperandPreparedFactory<BE>,
    Module<CIBackend<BE>>: CKKSEncodingOps<CIBackend<BE>, F>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE> + LWEInfos,
{
    let bytes = BootstrappingContextPrepared::<BE, F>::compile_unprepared_tmp_bytes(module, ci, plan).unwrap();
    let unprepared = with_scratch::<BE, _>(bytes, |s| {
        BootstrappingContextPrepared::<BE, F>::compile_unprepared(module, ci, base2k, plan, s)
    })
    .unwrap();
    let bytes = unprepared.prepare_tmp_bytes(module);
    with_scratch::<BE, _>(bytes, |s| unprepared.prepare(module, s)).unwrap()
}

/// [`BootstrappingContextPrepared::generate_keys`] of `layout` with EvalMod on the CI ring, on an
/// arena of exactly its query.
fn generate_ci_keys<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    context: &BootstrappingContextPrepared<BE, f64>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    layout: &BootstrappingKeysLayout,
) -> crate::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWECIKeyEncryptSk<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>,
{
    let bytes = layout.generate_tmp_bytes(module, Some(ci));
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
    with_scratch::<BE, _>(bytes, |s| {
        context.generate_keys(
            module,
            Some(ci),
            sk_raw,
            layout,
            &mut source_xs,
            &mut source_xe,
            &mut source_xa,
            s,
        )
    })
    .unwrap()
}

/// [`BootstrappingKeySet::prepare`](crate::layouts::BootstrappingKeySet::prepare) of `keys`, generated for
/// `layout` with EvalMod on the CI ring, on an arena of exactly its query.
fn prepare_keys_ci<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    keys: &crate::layouts::BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>,
    layout: &BootstrappingKeysLayout,
) -> crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE>
where
    BE: TestContextBackend<Ring = Standard> + CKKSCITwin,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>,
{
    with_scratch::<BE, _>(layout.prepare_tmp_bytes(module, Some(ci)), |s| {
        keys.prepare(module, Some(ci), s)
    })
    .unwrap()
}

/// The dense application secret at the preset's Hamming weight, raw and prepared.
fn dense_secret<BC: Backend>(
    client: &Module<BC>,
    preset: &BootstrappingPreset,
) -> (BackendGLWESecret<BC>, BackendGLWESecretPrepared<BC>)
where
    Module<BC>:
        ModuleCoreAlloc<OwnedBuf = BC::OwnedBuf, ZnxWord = BC::ZnxWord> + GLWESecretSampling<BC> + GLWESecretPreparedFactory<BC>,
{
    let sk_layout = preset.bootstrap_layout().glwe_layout;
    let mut sk_raw = client.glwe_secret_alloc_from_infos(&sk_layout);
    client.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
    let mut sk = client.glwe_secret_prepared_alloc_from_infos(&sk_layout);
    client.glwe_secret_prepare(&mut sk, &sk_raw);
    (sk_raw, sk)
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
    for atk in keys.rotation_key_layouts() {
        bytes = bytes.max(module.ckks_all_ops_with_atk_tmp_bytes(layout, &coeffs, &keys.tensor_key, atk));
    }
    bytes
}
