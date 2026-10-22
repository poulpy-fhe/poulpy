//! Full-slot CKKS bootstrapping benchmark.
//!
//! One benchmark per preset in [`poulpy_ckks::presets::bootstrapping::all`],
//! at an explicitly chosen fixture radix. Criterion's name filter selects a
//! preset by its name. The setup, bootstrap call, and precision measurement use
//! the shared
//! [`BootstrappingPresetRun`] driver, so the benchmark exercises exactly what
//! the precision pin test checks, EvalMod on the conjugate-invariant ring included.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, EvalModRing},
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::{
        helpers::{TestContextBackend, TestContextModule, TestContextSharedModule},
        presets::{BootstrappingPresetRun, preset_with_max_base2k},
    },
};
use poulpy_core::{
    GLWECIKeyEncryptSk,
    layouts::{
        GGLWEInfos, GLWECIKeyPreparedFactory, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::ScratchOwnedAlloc,
    layouts::{Backend, ConjugateInvariant, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
};

fn runner_ckks_bootstrapping<BE>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    preset: BootstrappingPreset,
    setup: impl Fn(BootstrappingPreset) -> BootstrappingPresetRun<BE>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let id = format!(
        "{}(base2k={},dsize={},dense_to_sparse_dsize={})",
        preset.name(),
        preset.base2k(),
        preset.key_dsize(),
        preset.dense_to_sparse_dsize()
    );
    let mut run = None;
    let mut precision = None;
    group.bench_function(BenchmarkId::from_parameter(&id), |bencher| {
        let run = run.get_or_insert_with(|| setup(preset.clone()));
        bencher.iter(|| run.bootstrap());
        precision = Some(run.precision());
    });
    if let Some((re, im)) = precision {
        let backend = backend_name::<BE>();
        println!(
            "PRECISION backend={backend} preset={id} re_avg={:.2}b re_min={:.2}b re_worst_idx={} re_worst_err={:.3e} im_avg={:.2}b im_min={:.2}b im_worst_idx={} im_worst_err={:.3e} advertised={}b",
            re.avg_log2_prec,
            re.min_log2_prec,
            re.worst_idx,
            re.worst_err,
            im.avg_log2_prec,
            im.min_log2_prec,
            im.worst_idx,
            im.worst_err,
            preset.log2_precision(),
        );
        let advertised = preset.log2_precision() as f64;
        assert!(
            re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
            "preset {id} advertises {advertised} bits of precision"
        );
    }
}

/// Exercises every preset at the caller's fixture radix limit, `BCI` running
/// EvalMod for the presets that put it on the conjugate-invariant ring, which
/// have no other radix: a fixture radix below theirs leaves them out.
///
/// `FIXTURE_BASE2K` selects the benchmark shape; it carries no failure-probability
/// guarantee. The registered FFT and NTT fixtures use 18 and 52 respectively.
pub fn bench_ckks_bootstrapping<BE, BCI, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend<Ring = Standard>,
    BCI: TestContextBackend<Ring = ConjugateInvariant>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTMatrixOps<BE, f64>
        + CKKSBootstrappingCIOps<BE, BCI>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>,
    Module<BCI>: TestContextSharedModule<BCI> + CKKSEncodingOps<BCI, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = backend_name::<BE>();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for preset in all().unwrap() {
        match preset.plan().eval_mod_ring() {
            EvalModRing::Standard => {
                let preset = preset_with_max_base2k(&preset, FIXTURE_BASE2K).unwrap();
                runner_ckks_bootstrapping(&mut group, preset, BootstrappingPresetRun::<BE>::setup);
            }
            _ if preset.base2k() > FIXTURE_BASE2K => {}
            _ => runner_ckks_bootstrapping(&mut group, preset, |preset| {
                BootstrappingPresetRun::<BE>::setup_ci::<BCI>(preset)
            }),
        }
    }
    group.finish();
}
