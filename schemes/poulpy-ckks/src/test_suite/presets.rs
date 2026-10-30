//! End-to-end driver for the bootstrapping presets.
//!
//! [`BootstrappingPresetRun`] sets a preset up exactly as an application would
//! (compiled context, generated keys, an encrypted reference vector), runs the
//! bootstrap, and measures the output precision. The benchmarks and the
//! precision pin test ([`bootstrapping_presets_meet_precision`]) both drive it,
//! so there is a single description of how a preset is exercised.

use crate::Scale;
use poulpy_core::{
    DiagonalProd, GLWECIKeyEncryptSk,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWECIKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory, TransferInto,
    },
    layouts::{
        BackendGLWESecret, BackendGLWESecretPrepared, Base2K, GGLWEInfos, GLWESecretPrepared, GLWETensorKeyPrepared,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, SetK, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, HostStaged, MaxBase2k, Module, ReaderFrom, ScratchArena, ScratchOwned, Standard,
        WriterTo,
    },
    source::Source,
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDFTOps, CKKSDecryptOps,
        CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSEncryptOps, CKKSEvalModOps,
    },
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeysLayout, BootstrappingPlan, CIConversion, CKKSCiphertextOwned,
        CKKSModuleAlloc, CKKSPlaintextOwned, EvalModRing,
    },
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::helpers::{
        PrecisionStats, TestContextBackend, TestContextModule, TestContextSharedModule, assert_canonical_at_k, ckks_spec,
        precision_stats, test_vector_1,
    },
    test_suite::parity::helpers::{poisoned_scratch, with_scratch},
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

/// `Module<BE>` capabilities of the backend a [`BootstrappingPresetRun`] evaluates on.
pub trait PresetRunModule<BE: Backend>:
    ModuleNew<BE>
    + ModuleSynchronize<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + CKKSModuleAlloc<BE>
    + CKKSAllOpsTmpBytes<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSBootstrappingOps<BE>
    + CKKSDFTOps<BE>
    + CKKSEvalModOps<BE>
    + CKKSDFTMatrixOps<BE, f64>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + GLWESwitchingKeyPreparedFactory<BE>
{
}

impl<BE: Backend, M> PresetRunModule<BE> for M where
    M: ModuleNew<BE>
        + ModuleSynchronize<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSModuleAlloc<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
{
}

/// The bootstrap of a preset run: its context and keys, and those of the CI ring.
type Bootstrap<BE> = Box<
    dyn FnMut(&Module<BE>, &mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> CKKSResult<()>,
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
}

impl<BE, BC> BootstrappingPresetRun<BE, BC>
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + 'static,
    BC: TestContextBackend<Ring = Standard>,
    Module<BE>: PresetRunModule<BE>,
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
    /// comes from `preset`.
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
        let (module, client) = (Module::<BE>::new(n as u64), Module::<BC>::new(n as u64));

        let mut client_scratch =
            ScratchOwned::<BC>::alloc(bootstrap_setup_tmp_bytes(&client, &bootstrap_layout, plan, &keys_layout));
        let (sk_raw, sk) = dense_secret(&client, &preset);
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let keys = plan
            .generate_keys(
                &client,
                &sk_raw,
                &keys_layout,
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut client_scratch.borrow(),
            )
            .unwrap();
        if roundtrip_keys {
            Self::roundtrip_keys(&client, &keys);
        }

        let context = compile_context::<BE, f64>(&module, base2k.into(), plan);
        let boot_scratch = module.ckks_bootstrap_tmp_bytes(&bootstrap_layout, &input_layout, &context, &keys_layout);
        let mut scratch = ScratchOwned::<BE>::alloc(keys_layout.prepare_tmp_bytes(&module));
        let keys = keys.transfer_and_prepare(&module, &mut scratch.borrow());

        let bootstrap: Bootstrap<BE> =
            Box::new(move |module, output, input, scratch| module.ckks_bootstrap(output, input, &context, &keys, scratch));
        Self::with_bootstrap(
            preset,
            module,
            poisoned_scratch::<BE>(boot_scratch),
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

    /// [`Self::setup`] for a preset with EvalMod on the conjugate-invariant ring, on the twins
    /// [`CIBackend`] of the evaluated backend and of the client.
    pub fn setup_ci(preset: BootstrappingPreset) -> Self
    where
        Module<BE>: CKKSBootstrappingCIOps<BE> + CKKSDFTOps<BE> + GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSModuleAlloc<CIBackend<BE>>
            + CKKSEncodingOps<CIBackend<BE>, f64>
            + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
        ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
        CIBackend<BC>: TestContextBackend,
        Module<BC>: CKKSBootstrappingCIOps<BC> + GLWECIKeyEncryptSk<BC>,
        Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    {
        Self::setup_ci_with_roundtrip(preset, false)
    }

    /// [`Self::setup_ci`], with every key round-tripped through its serialization when
    /// `roundtrip_keys`.
    fn setup_ci_with_roundtrip(preset: BootstrappingPreset, roundtrip_keys: bool) -> Self
    where
        Module<BE>: CKKSBootstrappingCIOps<BE> + CKKSDFTOps<BE> + GLWECIKeyPreparedFactory<BE>,
        Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSModuleAlloc<CIBackend<BE>>
            + CKKSEncodingOps<CIBackend<BE>, f64>
            + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
        ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
        CIBackend<BC>: TestContextBackend,
        Module<BC>: CKKSBootstrappingCIOps<BC> + GLWECIKeyEncryptSk<BC>,
        Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    {
        // One module of degree 2N serves the conversions and the stages of degree N.
        let n = preset.n() as u64;
        let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
        let (client, client_ci) = (Module::<BC>::new(2 * n), Module::<CIBackend<BC>>::new(n));
        let (sk_raw, sk) = dense_secret(&client, &preset);

        // Every step runs on arenas of exactly its own query.
        let keys_layout = preset.keys_layout();
        let (bytes, bytes_ci) = keys_layout.generate_ci_tmp_bytes(&client, &client_ci);
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let keys = preset
            .plan()
            .generate_keys_ci(
                &client,
                &client_ci,
                &sk_raw,
                keys_layout,
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                &mut ScratchOwned::<BC>::alloc(bytes).borrow(),
                &mut ScratchOwned::<CIBackend<BC>>::alloc(bytes_ci).borrow(),
            )
            .unwrap();
        if roundtrip_keys {
            Self::roundtrip_keys(&client, &keys);
        }
        let context = compile_ci(&module, &ci, &preset);
        let (bytes, bytes_ci) = keys_layout.prepare_ci_tmp_bytes(&module, &ci);
        let keys = keys
            .transfer_and_prepare_ci(
                &module,
                &ci,
                &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
                &mut ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci).borrow(),
            )
            .unwrap();
        let (layout, input_layout) = (preset.bootstrap_layout(), preset.input_layout());
        let (bytes, bytes_ci) = module.ckks_bootstrap_ci_tmp_bytes(&ci, &layout, &input_layout, &context, keys_layout);
        let mut scratch_ci = poisoned_scratch::<CIBackend<BE>>(bytes_ci);
        let bootstrap: Bootstrap<BE> = Box::new(move |module, output, input, scratch| {
            module.ckks_bootstrap_ci_into(&ci, output, input, &context, &keys, scratch, &mut scratch_ci.borrow())
        });
        let client_scratch = ScratchOwned::<BC>::alloc(bootstrap_setup_tmp_bytes(&client, &layout, preset.plan(), keys_layout));
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

    /// Encrypts the reference vector at the preset's input layout on the client,
    /// moves it to the evaluated backend, and runs one bootstrap to check the
    /// output width.
    fn with_bootstrap(
        preset: BootstrappingPreset,
        module: Module<BE>,
        scratch: ScratchOwned<BE>,
        client: Module<BC>,
        mut client_scratch: ScratchOwned<BC>,
        sk: GLWESecretPrepared<BC::OwnedBuf, BC>,
        bootstrap: Bootstrap<BE>,
    ) -> Self {
        let (n, base2k) = (preset.n(), preset.base2k());
        let (input_layout, bootstrap_layout) = (preset.input_layout(), preset.bootstrap_layout());
        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut input_pt = client.ckks_pt_vec_alloc(n / 2, base2k.into(), input_layout.k());
        input_pt.set_meta(input_layout.meta());
        client
            .ckks_encode_reim_into(&mut input_pt, &want_re, &want_im, &mut client_scratch.borrow())
            .unwrap();

        let mut client_input = client.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        let (mut input_xa, mut input_xe) = (Source::new([3; 32]), Source::new([4; 32]));
        client
            .ckks_encrypt_sk(
                &mut client_input,
                &input_pt,
                &sk,
                &mut input_xe,
                &mut input_xa,
                &mut client_scratch.borrow(),
            )
            .unwrap();
        let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        client_input.inner.transfer_into(&mut input.inner);
        input.set_meta(client_input.meta());
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
        (self.bootstrap)(&self.module, &mut self.output, &self.input, &mut self.scratch.borrow()).unwrap();
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
        let mut output_pt =
            self.client
                .ckks_pt_vec_alloc(self.preset.n() / 2, output.base2k(), (output.log_delta() + log_budget).into());
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
            .ckks_decode_reim_into(&output_pt, &mut got_re, &mut got_im, &mut self.client_scratch.borrow())
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
        // EvalRound+ leaves its rotation keys to the stages' lookups: the sparse-to-dense
        // key of the raise, checked at entry, refuses first.
        let refused = if self.preset.plan().coeffs_to_slots_bypass().is_some() {
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand { op: "ckks_bootstrap" | "ckks_bootstrap_ci_into", reason })
                    if reason.starts_with("the sparse-to-dense key takes")
            )
        } else {
            matches!(err.composition(), Some(CKKSCompositionError::MissingAutomorphismKey { .. }))
        };
        assert!(
            refused,
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
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSModuleAlloc<CIBackend<BE>>
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
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard>,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: PresetRunModule<BE> + CKKSDFTOps<BE> + CKKSBootstrappingCIOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSModuleAlloc<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + CKKSBootstrappingCIOps<BC> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::DiagonalProd<BE>,
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
            match ring {
                EvalModRing::Standard => BootstrappingPresetRun::<BE, BC>::setup_with_roundtrip(preset, roundtrip).check(backend),
                _ => BootstrappingPresetRun::<BE, BC>::setup_ci_with_roundtrip(preset, std::mem::take(&mut first_ci))
                    .check(backend),
            }
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
/// shapes and output precision, then the same preset with EvalMod on the
/// conjugate-invariant ring, on the twin [`CIBackend`]. Unlike [`bootstrapping_presets_meet_precision`],
/// it is small enough for a default test run, so the narrow conjugation and
/// SlotsToCoeffs keys the presets derive run end to end. Backends register it
/// when their radix takes the preset's 52 bits: a smaller fixture radix gives
/// every key one digit, hence one shape.
pub fn small_s2c_preset_meets_precision<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSModuleAlloc<CIBackend<BE>>
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
    // EvalMod on the CI ring, with `u` as a product and folded, then all of it with
    // EvalRound+: CoeffsToSlots at 30 bits feeds EvalMod, the bypass at 50 sets the precision.
    // The keys of the first run, conversion keys included, go through their serialization.
    let eval_round_plus = preset.with_eval_round_plus(30, 50).unwrap();
    BootstrappingPresetRun::<BE>::setup(eval_round_plus.clone()).check(backend);
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
            BootstrappingPresetRun::<BE>::setup_ci_with_roundtrip(
                preset.with_eval_mod_ring(ci).unwrap(),
                std::mem::take(&mut roundtrip),
            )
            .check(backend);
        }
    }
    // The transforms with the keys of power-of-two rotations, `u` folded on the CI ring.
    let powers_of_two = preset.with_power_of_two_keys().unwrap();
    BootstrappingPresetRun::<BE>::setup(powers_of_two.clone()).check(backend);
    let ci = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        post: CIConversion::Product { log_delta: 38 },
    };
    BootstrappingPresetRun::<BE>::setup_ci(powers_of_two.with_eval_mod_ring(ci).unwrap()).check(backend);
}

/// The insecure `N = 2^11` S2C-first preset of [`small_s2c_preset_meets_precision`] on an
/// evaluated backend `BE` whose keys and ciphertexts come from the client backend `BC`: once on
/// the standard ring, and once with EvalMod on the conjugate-invariant ring, `u` folded and the
/// keys of power-of-two rotations. Small enough for a default test run, it moves every kind of
/// key and the ciphertexts both ways between two backends.
pub fn small_s2c_preset_meets_precision_with_client<BE, BC>()
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard>,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: PresetRunModule<BE> + CKKSDFTOps<BE> + CKKSBootstrappingCIOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSModuleAlloc<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + CKKSBootstrappingCIOps<BC> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::DiagonalProd<BE>,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let preset = crate::presets::bootstrapping::insecure_n11_d35_k180_p20_s2c().unwrap();
    BootstrappingPresetRun::<BE, BC>::setup(preset.clone()).check(backend);
    let ci = EvalModRing::ConjugateInvariant {
        pre: CIConversion::Folded,
        post: CIConversion::Product { log_delta: 38 },
    };
    let folded = preset.with_power_of_two_keys().unwrap().with_eval_mod_ring(ci).unwrap();
    BootstrappingPresetRun::<BE, BC>::setup_ci(folded).check(backend);
}

/// Key generation from the plan alone ([`BootstrappingPlan::generate_keys`] and
/// [`BootstrappingPlan::generate_keys_ci`]) takes the keys the matrices of the compiled context
/// rotate with, Galois elements and first-factor rotations, and the context's own key
/// generation, from the same plan, the same keys, encapsulation and conversion keys alike, on the
/// insecure `N = 2^11` S2C-first preset, with the keys of power-of-two rotations, and with EvalMod
/// on the conjugate-invariant ring, `u` folded into CoeffsToSlots.
pub fn bootstrapping_plan_keys_match_context<BE>()
where
    BE: TestContextBackend<Ring = Standard> + HostStaged + MaxBase2k,
    CIBackend<BE>: TestContextBackend + MaxBase2k,
    Module<BE>: TestContextModule<BE>
        + PresetRunModule<BE>
        + CKKSDFTOps<BE>
        + CKKSBootstrappingCIOps<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: TestContextSharedModule<CIBackend<BE>>
        + CKKSModuleAlloc<CIBackend<BE>>
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
            BootstrappingContextPrepared::<BE, f64>::compile(&module, preset.base2k().into(), plan, &mut scratch.borrow())
                .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_context = context
            .generate_keys(&module, &sk_raw, layout, &mut xs, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_plan = plan
            .generate_keys(&module, &sk_raw, layout, &mut xs, &mut xe, &mut xa, &mut scratch.borrow())
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
    }
    for preset in [
        standard.with_eval_mod_ring(folded).unwrap(),
        powers_of_two.with_eval_mod_ring(folded).unwrap(),
    ] {
        let n = preset.n() as u64;
        let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
        let (sk_raw, _) = dense_secret(&module, &preset);
        let (plan, layout) = (preset.plan(), preset.keys_layout());
        let context = compile_ci(&module, &ci, &preset);
        let (bytes, bytes_ci) = layout.generate_ci_tmp_bytes(&module, &ci);
        let (mut scratch, mut scratch_ci) = (
            ScratchOwned::<BE>::alloc(bytes),
            ScratchOwned::<CIBackend<BE>>::alloc(bytes_ci),
        );
        let (mut xs, mut xa, mut xe) = sources();
        let from_context = context
            .generate_keys_ci(
                &module,
                &ci,
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
                &mut scratch_ci.borrow(),
            )
            .unwrap();
        let (mut xs, mut xa, mut xe) = sources();
        let from_plan = plan
            .generate_keys_ci(
                &module,
                &ci,
                &sk_raw,
                layout,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
                &mut scratch_ci.borrow(),
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
    }
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
    BE: TestContextBackend<Ring = Standard>,
    CIBackend<BE>: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
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
        pre: CIConversion::Product { log_delta: 46 },
        post: CIConversion::Product { log_delta: 38 },
    };
    let preset = standard.with_eval_mod_ring(ring).unwrap();
    let n = preset.n() as u64;
    let (module, ci) = (Module::<BE>::new(2 * n), Module::<CIBackend<BE>>::new(n));
    let sk_layout = preset.bootstrap_layout().glwe_layout;
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
    let context = compile_ci(&module, &ci, &preset);
    let compiled = {
        let bytes = bootstrap_setup_tmp_bytes(&module, &standard.bootstrap_layout(), standard.plan(), standard.keys_layout());
        with_scratch::<BE, _>(bytes, |s| {
            BootstrappingContextPrepared::<BE, f64>::compile(&module, preset.base2k().into(), standard.plan(), s)
        })
        .unwrap()
    };
    // Key generation leaves nothing of the CI secret, nor of the encryptions, in the CI scratch.
    let (bytes, bytes_ci) = preset.keys_layout().generate_ci_tmp_bytes(&module, &ci);
    poulpy_core::test_suite::assert_wipes_scratch::<CIBackend<BE>>(bytes_ci, |scratch_ci| {
        let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        with_scratch::<BE, _>(bytes, |s| {
            context.generate_keys_ci(
                &module,
                &ci,
                &sk_raw,
                preset.keys_layout(),
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                s,
                scratch_ci,
            )
        })
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
                            context: &BootstrappingContextPrepared<BE, f64>,
                            keys: &crate::layouts::BootstrappingKeysPrepared<BE::OwnedBuf, BE, CIBackend<BE>>,
                            (bytes, bytes_ci): (usize, usize)| {
        output.set_k(preset.bootstrap_k().into());
        let before = output.to_host_owned::<BE>();
        let result = with_scratch::<BE, _>(bytes, |s| {
            with_scratch::<CIBackend<BE>, _>(bytes_ci, |sc| {
                module.ckks_bootstrap_ci_into(&ci, &mut output, &input, context, keys, s, sc)
            })
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
    for short in [
        (bytes - BE::SCRATCH_ALIGN, bytes_ci),
        (bytes, bytes_ci - CIBackend::<BE>::SCRATCH_ALIGN),
    ] {
        let err = bootstrap_on(&module, &context, &keys, short).unwrap_err();
        assert!(matches!(err, crate::CKKSError::Internal(_)), "short arena: {err}");
    }
    let mut bootstrap =
        |module: &Module<BE>,
         context: &BootstrappingContextPrepared<BE, f64>,
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

/// [`BootstrappingContextPrepared::compile`] of `plan`, each stage on an arena of exactly its query.
pub(crate) fn compile_context<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    plan: &BootstrappingPlan,
) -> BootstrappingContextPrepared<BE, F>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE> + LWEInfos,
{
    let bytes = BootstrappingContextPrepared::<BE, F>::compile_unprepared_tmp_bytes(module, plan).unwrap();
    let unprepared = with_scratch::<BE, _>(bytes, |s| {
        BootstrappingContextPrepared::<BE, F>::compile_unprepared(module, base2k, plan, s)
    })
    .unwrap();
    let bytes = unprepared.prepare_tmp_bytes(module);
    with_scratch::<BE, _>(bytes, |s| unprepared.prepare(module, s)).unwrap()
}

/// [`BootstrappingContextPrepared::compile_ci`] of `preset`, each stage on arenas of exactly its query.
fn compile_ci<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    preset: &BootstrappingPreset,
) -> BootstrappingContextPrepared<BE, f64>
where
    BE: Backend<Ring = Standard>,
    Module<BE>:
        CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, f64> + CKKSDFTMatrixOps<BE, f64> + CKKSBootstrappingCIOps<BE> + CKKSDFTOps<BE>,
    Module<CIBackend<BE>>: CKKSModuleAlloc<CIBackend<BE>> + CKKSEncodingOps<CIBackend<BE>, f64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::DiagonalProd<BE>,
{
    let (bytes, bytes_ci) =
        BootstrappingContextPrepared::<BE, f64>::compile_ci_unprepared_tmp_bytes(module, ci, preset.plan()).unwrap();
    let unprepared = with_scratch::<BE, _>(bytes, |s| {
        with_scratch::<CIBackend<BE>, _>(bytes_ci, |sc| {
            BootstrappingContextPrepared::<BE, f64>::compile_ci_unprepared(
                module,
                ci,
                preset.base2k().into(),
                preset.plan(),
                s,
                sc,
            )
        })
    })
    .unwrap();
    let bytes = unprepared.prepare_tmp_bytes(module);
    with_scratch::<BE, _>(bytes, |s| unprepared.prepare(module, s)).unwrap()
}

/// [`BootstrappingContextPrepared::generate_keys_ci`] of `layout`, on arenas of exactly its query.
fn generate_keys_ci<BE>(
    module: &Module<BE>,
    ci: &Module<CIBackend<BE>>,
    context: &BootstrappingContextPrepared<BE, f64>,
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
    with_scratch::<BE, _>(bytes, |s| {
        with_scratch::<CIBackend<BE>, _>(bytes_ci, |sc| {
            context.generate_keys_ci(
                module,
                ci,
                sk_raw,
                layout,
                &mut source_xs,
                &mut source_xe,
                &mut source_xa,
                s,
                sc,
            )
        })
    })
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
    with_scratch::<BE, _>(bytes, |s| {
        with_scratch::<CIBackend<BE>, _>(bytes_ci, |sc| keys.prepare_ci(module, ci, s, sc))
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
    for atk in &keys.rotation_key_layouts() {
        bytes = bytes.max(module.ckks_all_ops_with_atk_tmp_bytes(layout, &coeffs, &keys.tensor_key, atk));
    }
    bytes
}
