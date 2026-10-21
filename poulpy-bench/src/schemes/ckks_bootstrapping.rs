//! Full-slot CKKS bootstrapping benchmark.
//!
//! One benchmark per preset in [`poulpy_ckks::presets::bootstrapping::all`],
//! at an explicitly chosen fixture radix, and per preset with EvalMod on the
//! conjugate-invariant ring in [`poulpy_ckks::presets::bootstrapping::all_ci`],
//! in the same group. Criterion's name filter selects a preset by its name. The
//! setup, bootstrap call, and precision measurement use the shared
//! [`BootstrappingPresetRun`] and [`BootstrappingCIPresetRun`] drivers, so the
//! benchmark exercises exactly what the precision pin tests check.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned},
    oep::CKKSEncapsulatedModUpImpl,
    presets::bootstrapping::{BootstrappingPreset, all, all_ci},
    test_suite::{
        helpers::{TestContextBackend, TestContextModule, TestContextSharedModule},
        presets::{BootstrappingCIPresetRun, BootstrappingPresetRun, preset_with_max_base2k},
    },
};
use poulpy_core::{
    GLWECIKeyEncryptSk,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWEToBackendRef, GLWECIKeyPreparedFactory, GLWESecretToBackendRef, GLWETensorKey,
        GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::ScratchOwnedAlloc,
    layouts::{Backend, ConjugateInvariant, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
};

fn runner_ckks_bootstrapping<BE>(group: &mut BenchmarkGroup<'_, WallTime>, preset: BootstrappingPreset)
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
        let run = run.get_or_insert_with(|| BootstrappingPresetRun::<BE>::setup(preset.clone()));
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

/// Exercises every preset at the caller's fixture radix limit.
///
/// `FIXTURE_BASE2K` selects the benchmark shape; it carries no failure-probability
/// guarantee. The registered FFT and NTT fixtures use 18 and 52 respectively.
pub fn bench_ckks_bootstrapping<BE, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
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
    let backend = backend_name::<BE>();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for preset in all().unwrap() {
        let preset = preset_with_max_base2k(&preset, FIXTURE_BASE2K).unwrap();
        runner_ckks_bootstrapping::<BE>(&mut group, preset);
    }
    group.finish();
}

/// Exercises every preset with EvalMod on the conjugate-invariant ring, `BE`
/// running the standard stages and `BCI` EvalMod, in the group of
/// [`bench_ckks_bootstrapping`]. These presets have no other radix: one whose
/// nominal radix exceeds `FIXTURE_BASE2K` is left out.
pub fn bench_ckks_bootstrapping_ci<BE, BCI, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
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
    ScratchOwned<BCI>: ScratchOwnedAlloc<BCI>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    BackendGLWESecret<BCI>: GLWESecretToBackendRef<BE>,
    GLWETensorKey<BE::OwnedBuf, BE::ZnxWord>: GGLWEToBackendRef<BCI>,
{
    let backend = backend_name::<BE>();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for preset in all_ci().unwrap() {
        let base = preset.base();
        if base.base2k() > FIXTURE_BASE2K {
            continue;
        }
        let id = format!(
            "{}(base2k={},dsize={},dense_to_sparse_dsize={})",
            preset.name(),
            base.base2k(),
            base.key_dsize(),
            base.dense_to_sparse_dsize()
        );
        let advertised = base.log2_precision();
        let mut run = None;
        let mut precision = None;
        group.bench_function(BenchmarkId::from_parameter(&id), |bencher| {
            let run = run.get_or_insert_with(|| BootstrappingCIPresetRun::<BE, BCI>::setup(preset.clone()));
            bencher.iter(|| run.bootstrap());
            precision = Some(run.precision());
        });
        if let Some((re, im)) = precision {
            println!(
                "PRECISION backend={backend} preset={id} re_avg={:.2}b re_min={:.2}b re_worst_idx={} re_worst_err={:.3e} im_avg={:.2}b im_min={:.2}b im_worst_idx={} im_worst_err={:.3e} advertised={advertised}b",
                re.avg_log2_prec,
                re.min_log2_prec,
                re.worst_idx,
                re.worst_err,
                im.avg_log2_prec,
                im.min_log2_prec,
                im.worst_idx,
                im.worst_err,
            );
            assert!(
                re.min_log2_prec >= advertised as f64 && im.min_log2_prec >= advertised as f64,
                "preset {id} advertises {advertised} bits of precision"
            );
        }
    }
    group.finish();
}
