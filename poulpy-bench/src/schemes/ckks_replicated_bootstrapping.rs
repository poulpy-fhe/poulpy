//! Bootstrapping of a replicated slot layout, with the parameters of
//! ePrint 2026/2513 ([`ReplicatedBootstrapPlan::paper`]).
//!
//! One benchmark per ring degree, batch size `k` and output levels `L`, with
//! the unit `U32_45` and the keys of
//! [`ReplicatedBootstrapping::secure_key_layouts`] within the 128-bit modulus
//! bounds of a dense secret of Hamming weight 1024 and an ephemeral secret of
//! weight 32. Criterion's name filter selects a configuration.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSLinearTransformationOps, CKKSModuleInfos,
        CKKSReplicatedBootstrapOps,
    },
    layouts::{
        BootstrappingKeysPrepared, CKKSCiphertextOwned, CKKSModuleAlloc, ReplicatedBootstrapPlan, ReplicatedBootstrapping,
    },
    oep::CKKSEncapsulatedModUpImpl,
    shaping::ShapingUnit,
    test_suite::helpers::{TestContextBackend, TestContextModule},
};
use poulpy_core::{
    GLWETrace,
    layouts::prepared::GLWESecretPreparedFactory,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWELayout, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
    source::Source,
};

const HAMMING_WEIGHT: usize = 1024;
const EPHEMERAL_HAMMING_WEIGHT: usize = 32;
/// Data-slot precision every configuration keeps, in bits.
const MIN_PRECISION: f64 = 19.0;

/// `(log2 N, k, L)`.
const CONFIGURATIONS: [(usize, usize, usize); 8] = [
    (14, 16, 1),
    (14, 16, 2),
    (14, 64, 1),
    (14, 64, 2),
    (15, 16, 1),
    (15, 16, 16),
    (15, 64, 1),
    (15, 64, 16),
];

/// 128-bit bounds on `log2(QP)` for secrets of Hamming weight 1024 and 32.
fn modulus_bounds(log_n: usize) -> (usize, usize) {
    match log_n {
        14 => (424, 77),
        15 => (851, 162),
        _ => unreachable!("no bound for log2 N = {log_n}"),
    }
}

/// What the benchmark asks of a module.
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

/// Keys, input and scratch of one configuration.
struct Run<BE: Backend> {
    module: Module<BE>,
    ctx: ReplicatedBootstrapping<BE, f64>,
    keys: BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    input: CKKSCiphertextOwned<BE>,
    output: CKKSCiphertextOwned<BE>,
    values: (Vec<f64>, Vec<f64>),
    scratch: ScratchOwned<BE>,
    /// The gadgets of the keys under the secret and of the key to the ephemeral secret.
    gadgets: String,
}

impl<BE> Run<BE>
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    fn setup(base2k: usize, log_n: usize, k: usize, levels: usize) -> Self {
        let n = 1usize << log_n;
        let module = Module::<BE>::new(n as u64);
        let layout = ShapingUnit::U32_45.layout(k.trailing_zeros() as u8, 24).unwrap();
        let plan = ReplicatedBootstrapPlan::paper(layout);
        let mut scratch = ScratchOwned::<BE>::alloc(1 << 28);
        let ctx = ReplicatedBootstrapping::<BE, f64>::new(&module, plan, base2k.into(), &mut scratch.borrow()).unwrap();
        let k_out = plan.k_in() + levels * plan.level_log_delta;
        let (max_dense, max_sparse) = modulus_bounds(log_n);
        let keys_layout = ctx
            .secure_key_layouts(n.into(), base2k.into(), k_out, max_dense, max_sparse)
            .unwrap();

        let glwe = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_out.into(),
            rank: Rank(1),
        };
        let (mut xs, mut xe, mut xa) = (Source::new([0; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, HAMMING_WEIGHT, &mut xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let keys = ctx
            .generate_keys(
                &module,
                &sk_raw,
                &keys_layout,
                EPHEMERAL_HAMMING_WEIGHT,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut scratch.borrow(),
            )
            .prepare(&module, &mut scratch.borrow());

        let values: (Vec<f64>, Vec<f64>) = (0..k).map(|i| (0.37 * (i + 1) as f64).sin_cos()).map(|(s, c)| (c, s)).unzip();
        let mut pt = module.ckks_pt_vec_alloc(k << layout.log_r(), base2k.into(), (plan.log_delta + 8).into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(plan.log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Replicated(layout),
        });
        module
            .ckks_encode_reim_into(&mut pt, &values.0, &values.1, &mut scratch.borrow())
            .unwrap();
        let mut input = module.ckks_ciphertext_alloc(base2k.into(), plan.k_in().into());
        module
            .ckks_encrypt_sk(&mut input, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())
            .unwrap();

        let gadget = |key: &GGLWELayout| format!("{}x{}+{}", key.dnum.0, key.k_digit.0, key.k_aux.0);
        let encapsulation = keys_layout.encapsulation.expect("the keys carry encapsulation");
        let gadgets = format!(
            "keys={} ephemeral={}",
            gadget(&keys_layout.automorphism_key.gglwe_layout()),
            gadget(&encapsulation.dense_to_sparse.gglwe_layout())
        );
        let output = module.ckks_ciphertext_alloc(base2k.into(), k_out.into());
        let encapsulation = keys_layout.encapsulation.map(|e| (e.dense_to_sparse, e.sparse_to_dense));
        let bytes = module.ckks_replicated_bootstrap_tmp_bytes(
            &output,
            &ctx,
            &keys_layout.automorphism_key,
            &keys_layout.tensor_key,
            encapsulation.as_ref().map(|(d, s)| (d, s)),
        );
        Self {
            module,
            ctx,
            keys,
            sk,
            input,
            output,
            values,
            scratch: ScratchOwned::<BE>::alloc(bytes.max(1 << 24)),
            gadgets,
        }
    }

    fn bootstrap(&mut self) {
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
    fn precision(&mut self) -> (f64, f64) {
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

fn runner<BE>(group: &mut BenchmarkGroup<'_, WallTime>, base2k: usize, log_n: usize, k: usize, levels: usize)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let id = format!("logn={log_n},k={k},levels={levels},base2k={base2k}");
    let mut run = None;
    let mut precision = None;
    group.bench_function(BenchmarkId::from_parameter(&id), |bencher| {
        let run = run.get_or_insert_with(|| Run::<BE>::setup(base2k, log_n, k, levels));
        bencher.iter(|| run.bootstrap());
        precision = Some((run.precision(), run.gadgets.clone()));
    });
    if let Some(((min, avg), gadgets)) = precision {
        println!(
            "PRECISION backend={} replicated {id} {gadgets} min={min:.2}b avg={avg:.2}b",
            backend_name::<BE>()
        );
        assert!(min >= MIN_PRECISION, "replicated bootstrapping {id}: {min:.2} bits");
    }
}

/// Bootstraps every configuration at the fixture radix `FIXTURE_BASE2K`.
pub fn bench_ckks_replicated_bootstrapping<BE, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let mut group = c.benchmark_group(format!("{}/ckks/ckks_replicated_bootstrapping", backend_name::<BE>()));
    group.sample_size(10);
    for (log_n, k, levels) in CONFIGURATIONS {
        runner::<BE>(&mut group, FIXTURE_BASE2K, log_n, k, levels);
    }
    group.finish();
}
