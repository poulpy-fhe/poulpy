//! Full-slot CKKS bootstrapping benchmark.
//!
//! One benchmark per preset in [`poulpy_ckks::presets::bootstrapping::all`],
//! re-derived at the backend's digit shape by
//! [`preset_for_backend`](poulpy_ckks::test_suite::presets::preset_for_backend).
//! Criterion's name filter selects a preset by its name. The setup, the
//! bootstrap call, and the precision measurement are the shared
//! [`BootstrappingPresetRun`] driver, so the benchmark exercises exactly what
//! the precision pin test checks.

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSEncodingOps},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned},
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::{
        helpers::{TestContextBackend, TestContextHostModule, TestContextModule},
        presets::{BootstrappingPresetRun, preset_for_backend},
    },
};
use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::ScratchOwnedAlloc,
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, Normalized, ScratchOwned},
};

fn runner_ckks_bootstrapping<BE>(group: &mut BenchmarkGroup<'_, WallTime>, preset: BootstrappingPreset)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>:
        GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized> + LWEInfos,
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
        let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
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
        if BE::DFT_IS_EXACT {
            let advertised = preset.log2_precision() as f64;
            assert!(
                re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
                "preset {id} advertises {advertised} bits of precision"
            );
        }
    }
}

pub fn bench_ckks_bootstrapping<BE>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, f64> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>:
        GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE, State = Normalized> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for preset in all().unwrap() {
        runner_ckks_bootstrapping::<BE>(&mut group, preset_for_backend::<BE>(&preset).unwrap());
    }
    group.finish();
}
