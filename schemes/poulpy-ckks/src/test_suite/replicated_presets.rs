//! End-to-end exercise of the replicated bootstrapping presets.
//!
//! [`ReplicatedBootstrappingPresetRun`] sets a preset up as an application
//! would (compiled bootstrapping, generated keys, a batch encrypted at the
//! leveled scale and the preset's input width), runs the bootstrap, and
//! measures the precision of the data slots. The benchmarks and the precision
//! pin test ([`replicated_bootstrapping_presets_meet_precision`]) both drive it.
use poulpy_core::GLWESecretPreparedFactory;
use poulpy_core::GLWESecretSampling;

use poulpy_core::{
    GLWETrace, TransferInto,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{GGLWEInfos, GGLWELayout, GLWELayout, GLWESecretAlloc, GLWESecretPrepared, LWEInfos, Rank},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostStaged, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{
    CKKSInfos, CKKSLayout, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSCopyOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSReplicatedBootstrapOps,
    },
    layouts::{
        BootstrappingKeysLayout, BootstrappingKeysPrepared, CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSPlaintextAlloc,
        ReplicatedBootstrapWorkspace, ReplicatedBootstrapping,
    },
    oep::CKKSEncapsulatedModUpImpl,
    presets::replicated_bootstrapping::{ReplicatedBootstrappingPreset, all},
    shaping::ShapingUnit,
    test_suite::helpers::{TestContextBackend, TestContextModule},
    test_suite::parity::helpers::with_scratch,
};

/// What a replicated bootstrapping preset asks of a module.
pub trait ReplicatedBootstrappingModule<BE: Backend>:
    TestContextModule<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSReplicatedBootstrapOps<BE>
    + CKKSLinearTransformationOps<BE>
    + CnvPVecAlloc<BE>
    + GLWETrace<BE>
{
}

impl<BE: Backend, M> ReplicatedBootstrappingModule<BE> for M where
    M: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
{
}

/// What the evaluated backend of a replicated preset run asks of a module: it
/// compiles the bootstrapping, prepares the keys and bootstraps.
pub trait ReplicatedBootstrappingDeviceModule<BE: Backend>:
    ModuleNew<BE>
    + CKKSCiphertextAlloc<BE>
    + CKKSCopyOps<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSReplicatedBootstrapOps<BE>
    + CKKSLinearTransformationOps<BE>
    + CnvPVecAlloc<BE>
    + GLWETrace<BE>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + GLWESwitchingKeyPreparedFactory<BE>
{
}

impl<BE: Backend, M> ReplicatedBootstrappingDeviceModule<BE> for M where
    M: ModuleNew<BE>
        + CKKSCiphertextAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
{
}

/// A preset set up end to end for a batch of `k` values, ready to bootstrap
/// repeatedly on `BE`. The client `BC` holds the secret: it generates the keys,
/// which `BE` prepares, encrypts the batch, which moves to `BE`, and decrypts
/// the output moved back.
pub struct ReplicatedBootstrappingPresetRun<BE: Backend, BC: Backend = BE> {
    preset: ReplicatedBootstrappingPreset,
    module: Module<BE>,
    client: Module<BC>,
    ctx: ReplicatedBootstrapping<BE, f64>,
    workspace: ReplicatedBootstrapWorkspace<BE>,
    keys: BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    sk: GLWESecretPrepared<BC::OwnedBuf, BC>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    values: (Vec<f64>, Vec<f64>),
    scratch: ScratchOwned<BE>,
    client_scratch: ScratchOwned<BC>,
}

impl<BE, BC> ReplicatedBootstrappingPresetRun<BE, BC>
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + CKKSEncapsulatedModUpImpl + 'static,
    Module<BE>: ReplicatedBootstrappingDeviceModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    BC: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BC>: ReplicatedBootstrappingModule<BC>,
{
    /// `k` values with the unit `U32_45`.
    pub fn setup(preset: ReplicatedBootstrappingPreset, k: usize) -> Self {
        Self::setup_with_keys(preset, k, |keys_layout| keys_layout)
    }

    /// [`Self::setup`] with the preset's key layouts passed through `adjust`.
    fn setup_with_keys(
        preset: ReplicatedBootstrappingPreset,
        k: usize,
        adjust: impl FnOnce(BootstrappingKeysLayout) -> BootstrappingKeysLayout,
    ) -> Self {
        let n = preset.n();
        let base2k = preset.base2k();
        let (module, client) = (Module::<BE>::new(n as u64), Module::<BC>::new(n as u64));
        let layout = ShapingUnit::U32_45.layout(k.trailing_zeros() as u8, 24).unwrap();
        let plan = preset.plan(layout);
        let ctx = with_scratch::<BE, _>(
            ReplicatedBootstrapping::<BE, f64>::new_tmp_bytes(&module, &plan, base2k.into()).unwrap(),
            |s| ReplicatedBootstrapping::<BE, f64>::new(&module, plan, base2k.into(), s),
        )
        .unwrap();
        // The client compiles the same plan, for the Galois elements and the keys of its secret.
        let client_ctx = with_scratch::<BC, _>(
            ReplicatedBootstrapping::<BC, f64>::new_tmp_bytes(&client, &plan, base2k.into()).unwrap(),
            |s| ReplicatedBootstrapping::<BC, f64>::new(&client, plan, base2k.into(), s),
        )
        .unwrap();
        let galois_elements = ctx.galois_elements(&module);
        assert_eq!(client_ctx.galois_elements(&client), galois_elements);
        let keys_layout = adjust(preset.keys_layout(&galois_elements));
        let input_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: preset.input_k().into(),
                rank: Rank(1),
            },
            meta: preset.output_layout(layout).meta,
        };
        let slots = k << layout.log_r();
        // The client's arena, sized from the queries of the calls it serves: key generation,
        // encoding and encryption of the batch, decryption and decoding of the output.
        let output_layout = preset.output_layout(layout);
        let decrypted = |degree: usize| CKKSLayout {
            glwe_layout: GLWELayout {
                n: degree.into(),
                base2k: base2k.into(),
                k: (plan.level_log_delta + 10).into(),
                rank: Rank(1),
            },
            meta: output_layout.meta,
        };
        let mut client_scratch = ScratchOwned::<BC>::alloc(
            client_ctx
                .generate_keys_tmp_bytes(&client, &keys_layout)
                .max(client.ckks_encoding_host_tmp_bytes(slots))
                .max(client.ckks_encrypt_sk_tmp_bytes(&input_layout))
                // `precision` decrypts at the degree of the output's sparsity: the ring's or the slots'.
                .max(client.ckks_decrypt_tmp_bytes(&decrypted(n), &output_layout))
                .max(client.ckks_decrypt_tmp_bytes(&decrypted(2 * slots), &output_layout)),
        );
        let glwe = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: preset.output_k().into(),
            rank: Rank(1),
        };
        let (mut xs, mut xe, mut xa) = (Source::new([0; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let mut sk_raw = client.glwe_secret_alloc_from_infos(&glwe);
        client.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut xs);
        let mut sk = client.glwe_secret_prepared_alloc_from_infos(&glwe);
        client.glwe_secret_prepare(&mut sk, &sk_raw);
        let keys = client_ctx.generate_keys(
            &client,
            &sk_raw,
            &keys_layout,
            preset.sparse_secret_hamming_weight(),
            &mut xs,
            &mut xe,
            &mut xa,
            &mut client_scratch.borrow(),
        );
        let keys = with_scratch::<BE, _>(keys_layout.prepare_tmp_bytes(&module), |s| {
            keys.transfer_and_prepare(&module, s)
        });

        let values: (Vec<f64>, Vec<f64>) = (0..k).map(|i| (0.37 * (i + 1) as f64).sin_cos()).map(|(s, c)| (c, s)).unzip();
        let mut pt = client.ckks_plaintext_alloc_from_slots(slots, base2k.into(), (plan.level_log_delta + 8).into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(plan.level_log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Replicated(layout),
        });
        client
            .ckks_encode_slots_host_into(&mut pt, &values.0, &values.1, &mut client_scratch.borrow())
            .unwrap();
        let mut client_input = client.ckks_ciphertext_alloc_from_infos(&input_layout);
        client
            .ckks_encrypt_sk(&mut client_input, &pt, &sk, &mut xe, &mut xa, &mut client_scratch.borrow())
            .unwrap();
        let mut input = module.ckks_ciphertext_alloc_from_infos(&input_layout);
        client_input.inner.transfer_into(&mut input.inner);
        input.set_meta(client_input.meta());

        let output = module.ckks_ciphertext_alloc_from_infos(&output_layout);
        let workspace = ctx.alloc_workspace(&module, &output);
        let encapsulation = keys_layout.encapsulation.map(|e| (e.dense_to_sparse, e.sparse_to_dense));
        // The bootstrapping and the copy of an input.
        let run_bytes = module
            .ckks_replicated_bootstrap_tmp_bytes(
                &output,
                &input,
                &ctx,
                keys_layout.rotation_keys.values().next().expect("the bootstrapping rotates"),
                &keys_layout.tensor_key,
                encapsulation.as_ref().map(|(d, s)| (d, s.as_ref().unwrap())),
            )
            .max(module.ckks_copy_tmp_bytes(&input, &input));
        Self {
            preset,
            module,
            client,
            ctx,
            workspace,
            keys,
            sk,
            input,
            output,
            values,
            scratch: ScratchOwned::<BE>::alloc(run_bytes),
            client_scratch,
        }
    }

    pub fn preset(&self) -> &ReplicatedBootstrappingPreset {
        &self.preset
    }

    /// The gadgets of the keys under the secret and of the key to the
    /// ephemeral secret, as `dnum x k_digit + k_aux`.
    pub fn gadgets(&self) -> String {
        let gadget = |key: GGLWELayout| format!("{}x{}+{}", key.dnum.0, key.k_digit.0, key.k_aux.0);
        let layout = self.preset.keys_layout(&[]);
        let ephemeral = layout.encapsulation.map(|e| gadget(e.dense_to_sparse.gglwe_layout()));
        let return_key = layout
            .encapsulation
            .and_then(|e| e.sparse_to_dense)
            .map(|key| format!("{}+{}", key.k_offset.0, gadget(key.gglwe_layout())));
        format!(
            "keys={} ephemeral={} return={}",
            gadget(layout.tensor_key.gglwe_layout()),
            ephemeral.unwrap_or_default(),
            return_key.unwrap_or_default()
        )
    }

    pub fn bootstrap(&mut self) {
        self.module
            .ckks_replicated_bootstrap(
                &mut self.output,
                &self.input,
                &self.ctx,
                &mut self.workspace,
                &self.keys,
                &mut self.scratch.borrow(),
            )
            .unwrap();
    }

    /// `(min, avg)` bits over the real and imaginary parts of the data values, the
    /// output moved to the client and decrypted there.
    pub fn precision(&mut self) -> (f64, f64) {
        let mut out = self.client.ckks_ciphertext_alloc_from_infos(&self.output);
        self.output.inner.transfer_into(&mut out.inner);
        out.set_meta(self.output.meta());
        let degree = out.n().as_usize() >> GLWEPlaintextInfos::log_sparsity(&out);
        let mut pt = self.client.ckks_plaintext_alloc_from_slots(
            self.client.ckks_slots_of_degree(degree),
            out.base2k(),
            (out.log_delta() + 10).into(),
        );
        pt.set_meta(out.meta());
        pt.set_log_budget(10);
        self.client
            .ckks_decrypt(&mut pt, &out, &self.sk, &mut self.client_scratch.borrow())
            .unwrap();
        let k = self.values.0.len();
        let (mut re, mut im) = (vec![0.0; k], vec![0.0; k]);
        self.client
            .ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut self.client_scratch.borrow())
            .unwrap();
        let errors: Vec<f64> = (0..k)
            .flat_map(|i| [(re[i] - self.values.0[i]).abs(), (im[i] - self.values.1[i]).abs()])
            .collect();
        let max = errors.iter().copied().fold(0.0, f64::max);
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        (-max.log2(), -mean.log2())
    }
}

/// Batch sizes the presets are exercised with.
pub const BATCH_SIZES: [usize; 2] = [16, 64];

/// Every replicated bootstrapping preset, at limb radix `fixture_base2k` when
/// smaller, meets its advertised precision on the data slots for every batch
/// size of [`BATCH_SIZES`]. Full-size bootstraps are slow, so backends register
/// this as an ignored test.
pub fn replicated_bootstrapping_presets_meet_precision<BE>(fixture_base2k: usize)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE> + ReplicatedBootstrappingDeviceModule<BE>,
{
    replicated_bootstrapping_presets_meet_precision_with_client::<BE, BE>(fixture_base2k)
}

/// [`replicated_bootstrapping_presets_meet_precision`] evaluated on `BE`, with the
/// keys generated, the batch encrypted and the output decrypted on the client `BC`.
pub fn replicated_bootstrapping_presets_meet_precision_with_client<BE, BC>(fixture_base2k: usize)
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + CKKSEncapsulatedModUpImpl + 'static,
    Module<BE>: ReplicatedBootstrappingDeviceModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    BC: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BC>: ReplicatedBootstrappingModule<BC>,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    for preset in all().unwrap() {
        let preset = preset.with_base2k(preset.base2k().min(fixture_base2k)).unwrap();
        for k in BATCH_SIZES {
            let check = |run: &mut ReplicatedBootstrappingPresetRun<BE, BC>, keys: &str| {
                run.bootstrap();
                let (min, avg) = run.precision();
                println!(
                    "PRECISION backend={backend} preset={} k={k} {keys} min={min:.2}b avg={avg:.2}b (advertised {}b)",
                    preset.name(),
                    preset.log2_precision()
                );
                assert!(
                    min >= preset.log2_precision() as f64,
                    "preset {} with k = {k} and {keys} advertises {} bits but measured {min:.2}",
                    preset.name(),
                    preset.log2_precision()
                );
            };
            let mut run = ReplicatedBootstrappingPresetRun::<BE, BC>::setup(preset.clone(), k);
            check(&mut run, "offset");

            let k_boot = run.ctx.plan().k_boot(preset.output_k());
            let mut full = ReplicatedBootstrappingPresetRun::<BE, BC>::setup_with_keys(preset.clone(), k, |mut keys_layout| {
                let wide = keys_layout.tensor_key;
                let return_key = keys_layout.encapsulation.as_mut().unwrap().sparse_to_dense.as_mut().unwrap();
                return_key.k_offset = 0usize.into();
                return_key.k_digit = wide.k_digit;
                return_key.k_aux = wide.k_aux;
                return_key.dnum = GGLWELayout::dnum_for_input(k_boot.into(), wide.k_digit);
                keys_layout
            });
            check(&mut full, "full-width");
            run.module
                .ckks_copy(&mut run.input, &full.input, &mut run.scratch.borrow())
                .unwrap();
            check(&mut run, "offset/shared-input");
        }
    }
}
