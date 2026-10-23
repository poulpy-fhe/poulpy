//! End-to-end exercise of the replicated bootstrapping presets.
//!
//! [`ReplicatedBootstrappingPresetRun`] sets a preset up as an application
//! would (compiled bootstrapping, generated keys, a batch encrypted at the
//! leveled scale and the preset's input width), runs the bootstrap, and
//! measures the precision of the data slots. The benchmarks and the precision
//! pin test ([`replicated_bootstrapping_presets_meet_precision`]) both drive it.

use poulpy_core::{
    GLWETrace,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWELayout, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos,
        ModuleCoreAlloc, Rank,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSCopyOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSReplicatedBootstrapOps,
    },
    layouts::{
        BootstrappingKeysLayout, BootstrappingKeysPrepared, CKKSCiphertextOwned, CKKSModuleAlloc, ReplicatedBootstrapping,
    },
    oep::CKKSEncapsulatedModUpImpl,
    presets::replicated_bootstrapping::{ReplicatedBootstrappingPreset, all},
    shaping::ShapingUnit,
    test_suite::helpers::{TestContextBackend, TestContextModule},
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

/// A preset set up end to end for a batch of `k` values, ready to bootstrap
/// repeatedly.
pub struct ReplicatedBootstrappingPresetRun<BE: Backend> {
    preset: ReplicatedBootstrappingPreset,
    module: Module<BE>,
    ctx: ReplicatedBootstrapping<BE, f64>,
    keys: BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    values: (Vec<f64>, Vec<f64>),
    scratch: ScratchOwned<BE>,
}

impl<BE> ReplicatedBootstrappingPresetRun<BE>
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    /// `k` values with the unit `U32_45`.
    pub fn setup(preset: ReplicatedBootstrappingPreset, k: usize) -> Self {
        let keys_layout = preset.keys_layout().clone();
        Self::setup_with_keys(preset, k, keys_layout)
    }

    fn setup_with_keys(preset: ReplicatedBootstrappingPreset, k: usize, keys_layout: BootstrappingKeysLayout) -> Self {
        let n = preset.n();
        let base2k = preset.base2k();
        let module = Module::<BE>::new(n as u64);
        let layout = ShapingUnit::U32_45.layout(k.trailing_zeros() as u8, 24).unwrap();
        let plan = preset.plan(layout);
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 28);
        let ctx = ReplicatedBootstrapping::<BE, f64>::new(&module, plan, base2k.into(), &mut scratch.borrow()).unwrap();
        let glwe = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: preset.output_k().into(),
            rank: Rank(1),
        };
        let (mut xs, mut xe, mut xa) = (Source::new([0; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let keys = ctx
            .generate_keys(
                &module,
                &sk_raw,
                &keys_layout,
                preset.sparse_secret_hamming_weight(),
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .prepare(&module, &mut scratch.borrow());

        let values: (Vec<f64>, Vec<f64>) = (0..k).map(|i| (0.37 * (i + 1) as f64).sin_cos()).map(|(s, c)| (c, s)).unzip();
        let mut pt = module.ckks_pt_vec_alloc(k << layout.log_r(), base2k.into(), (plan.level_log_delta + 8).into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(plan.level_log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Replicated(layout),
        });
        module
            .ckks_encode_reim_into(&mut pt, &values.0, &values.1, &mut scratch.borrow())
            .unwrap();
        let mut input = module.ckks_ciphertext_alloc(base2k.into(), preset.input_k().into());
        module
            .ckks_encrypt_sk(&mut input, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();

        let output = module.ckks_ciphertext_alloc_from_infos(&preset.output_layout(layout));
        let encapsulation = keys_layout.encapsulation.map(|e| (e.dense_to_sparse, e.sparse_to_dense));
        let bytes = module.ckks_replicated_bootstrap_tmp_bytes(
            &output,
            &input,
            &ctx,
            &keys_layout.automorphism_key,
            &keys_layout.tensor_key,
            encapsulation.as_ref().map(|(d, s)| (d, s.as_ref().unwrap())),
        );
        Self {
            preset,
            module,
            ctx,
            keys,
            sk,
            input,
            output,
            values,
            scratch: ScratchOwned::<BE>::alloc(bytes.max(1 << 24)),
        }
    }

    pub fn preset(&self) -> &ReplicatedBootstrappingPreset {
        &self.preset
    }

    /// The gadgets of the keys under the secret and of the key to the
    /// ephemeral secret, as `dnum x k_digit + k_aux`.
    pub fn gadgets(&self) -> String {
        let gadget = |key: GGLWELayout| format!("{}x{}+{}", key.dnum.0, key.k_digit.0, key.k_aux.0);
        let layout = self.preset.keys_layout();
        let ephemeral = layout.encapsulation.map(|e| gadget(e.dense_to_sparse.gglwe_layout()));
        let return_key = layout
            .encapsulation
            .and_then(|e| e.sparse_to_dense)
            .map(|key| format!("{}+{}", key.k_offset.0, gadget(key.gglwe_layout())));
        format!(
            "keys={} ephemeral={} return={}",
            gadget(layout.automorphism_key.gglwe_layout()),
            ephemeral.unwrap_or_default(),
            return_key.unwrap_or_default()
        )
    }

    pub fn bootstrap(&mut self) {
        self.module
            .ckks_replicated_bootstrap_into(
                &mut self.output,
                &self.input,
                &self.ctx,
                &self.keys,
                &mut self.scratch.borrow(),
            )
            .unwrap();
    }

    /// `(min, avg)` bits over the real and imaginary parts of the data values.
    pub fn precision(&mut self) -> (f64, f64) {
        let out = &self.output;
        let degree = out.n().as_usize() >> GLWEPlaintextInfos::log_sparsity(out);
        let mut pt = self.module.ckks_pt_vec_alloc(
            self.module.ckks_slots_of_degree(degree),
            out.base2k(),
            (out.log_delta() + 10).into(),
        );
        pt.set_meta(out.meta());
        pt.set_log_budget(10);
        self.module
            .ckks_decrypt(&mut pt, out, &self.sk, &mut self.scratch.borrow())
            .unwrap();
        let k = self.values.0.len();
        let (mut re, mut im) = (vec![0.0; k], vec![0.0; k]);
        self.module
            .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.scratch.borrow())
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
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    for preset in all().unwrap() {
        let preset = preset.with_base2k(preset.base2k().min(fixture_base2k)).unwrap();
        for k in BATCH_SIZES {
            let check = |run: &mut ReplicatedBootstrappingPresetRun<BE>, keys: &str| {
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
            let mut run = ReplicatedBootstrappingPresetRun::<BE>::setup(preset.clone(), k);
            check(&mut run, "offset");

            let mut keys_layout = preset.keys_layout().clone();
            let wide = keys_layout.automorphism_key;
            let return_key = keys_layout.encapsulation.as_mut().unwrap().sparse_to_dense.as_mut().unwrap();
            return_key.k_offset = 0usize.into();
            return_key.k_digit = wide.k_digit;
            return_key.k_aux = wide.k_aux;
            return_key.dnum = GGLWELayout::dnum_for_input(run.ctx.plan().k_boot(preset.output_k()).into(), wide.k_digit);
            let mut full = ReplicatedBootstrappingPresetRun::<BE>::setup_with_keys(preset.clone(), k, keys_layout);
            check(&mut full, "full-width");
            run.module
                .ckks_copy(&mut run.input, &full.input, &mut run.scratch.borrow())
                .unwrap();
            check(&mut run, "offset/shared-input");
        }
    }
}
