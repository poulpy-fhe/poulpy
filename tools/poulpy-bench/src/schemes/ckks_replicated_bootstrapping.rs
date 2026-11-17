//! Bootstrapping of a replicated slot layout.
//!
//! One benchmark per preset in
//! [`poulpy_ckks::presets::replicated_bootstrapping::all`] and batch size of
//! [`BATCH_SIZES`], and one per map kind of the logN16 preset and batch size of
//! [`MAPS_BATCH_SIZES`], at an explicitly chosen fixture radix. Criterion's name
//! filter selects a configuration. The setup, bootstrap call and precision
//! measurement use the shared [`ReplicatedBootstrappingPresetRun`] driver, so
//! the benchmark exercises exactly what the precision pin test checks.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    layouts::ReplicatedTransforms,
    oep::CKKSEncapsulatedModUpImpl,
    presets::replicated_bootstrapping::{ReplicatedBootstrappingPreset, all},
    test_suite::{
        helpers::TestContextBackend,
        replicated_presets::{
            BATCH_SIZES, MAPS_BATCH_SIZES, ReplicatedBootstrappingDeviceModule, ReplicatedBootstrappingModule,
            ReplicatedBootstrappingPresetRun,
        },
    },
};
use poulpy_hal::{
    api::ScratchOwnedAlloc,
    layouts::{Backend, HostStaged, Module, ScratchOwned},
};

fn runner<BE, BC>(group: &mut BenchmarkGroup<'_, WallTime>, preset: ReplicatedBootstrappingPreset, k: usize)
where
    BE: Backend<ZnxWord = i64> + HostStaged + CKKSEncapsulatedModUpImpl + 'static,
    Module<BE>: ReplicatedBootstrappingDeviceModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    BC: TestContextBackend + CKKSEncapsulatedModUpImpl,
    Module<BC>: ReplicatedBootstrappingModule<BC>,
{
    let maps = match preset.transforms() {
        ReplicatedTransforms::Dense => "dense",
        _ => "factored",
    };
    let id = format!("{}({maps},k={k},base2k={})", preset.name(), preset.base2k());
    let backend = backend_name::<BE>();
    let mut run = None;
    group.bench_function(BenchmarkId::from_parameter(&id), |bencher| {
        let run = run.get_or_insert_with(|| {
            let mut run = ReplicatedBootstrappingPresetRun::<BE, BC>::setup(preset.clone(), k);
            // The first call is verified before any is timed.
            run.bootstrap();
            let (min, avg) = run.precision();
            let advertised = preset.log2_precision() as f64;
            println!(
                "PRECISION backend={backend} preset={id} {} rotation_keys={} raised_k={} min={min:.2}b avg={avg:.2}b advertised={advertised}b",
                run.gadgets(),
                run.rotation_keys(),
                run.raised_k()
            );
            assert!(
                min >= advertised,
                "preset {id} advertises {advertised} bits but measured {min:.2}"
            );
            run
        });
        bencher.iter(|| run.bootstrap());
    });
}

/// Exercises every preset and batch size at the caller's fixture radix limit.
pub fn bench_ckks_replicated_bootstrapping<BE, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend + HostStaged + CKKSEncapsulatedModUpImpl + 'static,
    Module<BE>: ReplicatedBootstrappingModule<BE> + ReplicatedBootstrappingDeviceModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    bench_ckks_replicated_bootstrapping_with_client::<BE, BE, FIXTURE_BASE2K>(c)
}

/// [`bench_ckks_replicated_bootstrapping`] for an evaluated backend `BE` whose
/// keys and ciphertexts come from the client backend `BC`. The timed region is
/// the bootstrap on `BE`, synchronized.
pub fn bench_ckks_replicated_bootstrapping_with_client<BE, BC, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: Backend<ZnxWord = i64> + HostStaged + CKKSEncapsulatedModUpImpl + 'static,
    Module<BE>: ReplicatedBootstrappingDeviceModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    BC: TestContextBackend + CKKSEncapsulatedModUpImpl,
    Module<BC>: ReplicatedBootstrappingModule<BC>,
{
    let mut group = c.benchmark_group(format!("{}/ckks/ckks_replicated_bootstrapping", backend_name::<BE>()));
    group.sample_size(10);
    for preset in all().unwrap() {
        let preset = preset.with_base2k(preset.base2k().min(FIXTURE_BASE2K)).unwrap();
        for k in BATCH_SIZES {
            runner::<BE, BC>(&mut group, preset.clone(), k);
        }
        if preset.log_n() == 16 {
            for transforms in [preset.transforms(), ReplicatedTransforms::Dense] {
                let preset = preset.with_transforms(transforms).unwrap();
                for k in MAPS_BATCH_SIZES {
                    runner::<BE, BC>(&mut group, preset.clone(), k);
                }
            }
        }
    }
    group.finish();
}
