//! Full-slot CKKS bootstrapping benchmark.
//!
//! One benchmark per preset in [`poulpy_ckks::presets::bootstrapping::all`],
//! at the radix the backend's failure model selects. Criterion's name filter selects a
//! preset by its name. The setup, bootstrap call, and precision measurement use
//! the shared
//! [`BootstrappingPresetRun`] driver, so the benchmark exercises exactly what
//! the precision pin test checks, EvalMod on the conjugate-invariant ring included.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSDFTOps, CKKSEncodingOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EvalModRing},
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::{
        helpers::{TestContextBackend, TestContextModule, TestContextSharedModule},
        presets::{BootstrappingPresetRun, CIBackend, PresetRunModule, preset_for_backend},
    },
};
use poulpy_core::{
    GLWECIKeyEncryptSk,
    api::{GLWECIKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc},
    layouts::{Backend, HostDataMut, HostDataRef, HostStaged, MaxBase2k, Module, ScratchOwned, Standard},
};

fn runner_ckks_bootstrapping<BE, BC>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    preset: BootstrappingPreset,
    setup: impl Fn(BootstrappingPreset) -> BootstrappingPresetRun<BE, BC>,
) where
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

/// Exercises every preset at the radix
/// [`preset_for_backend`](poulpy_ckks::test_suite::presets::preset_for_backend)
/// selects for `BE`, the twin [`CIBackend`] running EvalMod for the presets that
/// put it on the conjugate-invariant ring.
pub fn bench_ckks_bootstrapping<BE>(c: &mut Criterion<WallTime>)
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
    bench_ckks_bootstrapping_with_client::<BE, BE>(c)
}

/// [`bench_ckks_bootstrapping`] for an evaluated backend `BE` whose keys and
/// ciphertexts come from the client backend `BC`, each with its conjugate-invariant
/// twin [`CIBackend`]. The timed region is the bootstrap on `BE`, synchronized.
pub fn bench_ckks_bootstrapping_with_client<BE, BC>(c: &mut Criterion<WallTime>)
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
    let backend = backend_name::<BE>();
    let mut group = c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping"));
    group.sample_size(10);
    for preset in all().unwrap() {
        let ring = preset.plan().eval_mod_ring();
        let preset = preset_for_backend::<BE, CIBackend<BE>>(&preset).unwrap();
        match ring {
            EvalModRing::Standard => runner_ckks_bootstrapping(&mut group, preset, BootstrappingPresetRun::<BE, BC>::setup),
            _ => runner_ckks_bootstrapping(&mut group, preset, |preset| {
                BootstrappingPresetRun::<BE, BC>::setup_ci(preset)
            }),
        }
    }
    group.finish();
}
