//! Bootstrapping of a replicated slot layout.
//!
//! One benchmark per preset in
//! [`poulpy_ckks::presets::replicated_bootstrapping::all`] and batch size of
//! [`BATCH_SIZES`], at an explicitly chosen fixture radix. Criterion's name
//! filter selects a configuration. The setup, bootstrap call and precision
//! measurement use the shared [`ReplicatedBootstrappingPresetRun`] driver, so
//! the benchmark exercises exactly what the precision pin test checks.

use crate::backend_name;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use poulpy_ckks::{
    oep::CKKSEncapsulatedModUpImpl,
    presets::replicated_bootstrapping::{ReplicatedBootstrappingPreset, all},
    test_suite::{
        helpers::TestContextBackend,
        replicated_presets::{BATCH_SIZES, ReplicatedBootstrappingModule, ReplicatedBootstrappingPresetRun},
    },
};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, Module, Standard};

fn runner<BE>(group: &mut BenchmarkGroup<'_, WallTime>, preset: ReplicatedBootstrappingPreset, k: usize)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let id = format!("{}(k={k},base2k={})", preset.name(), preset.base2k());
    let advertised = preset.log2_precision() as f64;
    let mut run = None;
    let mut measured = None;
    group.bench_function(BenchmarkId::from_parameter(&id), |bencher| {
        let run = run.get_or_insert_with(|| ReplicatedBootstrappingPresetRun::<BE>::setup(preset.clone(), k));
        bencher.iter(|| run.bootstrap());
        measured = Some((run.precision(), run.gadgets()));
    });
    if let Some(((min, avg), gadgets)) = measured {
        println!(
            "PRECISION backend={} preset={id} {gadgets} min={min:.2}b avg={avg:.2}b advertised={advertised}b",
            backend_name::<BE>()
        );
        assert!(
            min >= advertised,
            "preset {id} advertises {advertised} bits but measured {min:.2}"
        );
    }
}

/// Exercises every preset and batch size at the caller's fixture radix limit.
pub fn bench_ckks_replicated_bootstrapping<BE, const FIXTURE_BASE2K: usize>(c: &mut Criterion<WallTime>)
where
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    Module<BE>: ReplicatedBootstrappingModule<BE>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let mut group = c.benchmark_group(format!("{}/ckks/ckks_replicated_bootstrapping", backend_name::<BE>()));
    group.sample_size(10);
    for preset in all().unwrap() {
        let preset = preset.with_base2k(preset.base2k().min(FIXTURE_BASE2K)).unwrap();
        for k in BATCH_SIZES {
            runner::<BE>(&mut group, preset.clone(), k);
        }
    }
    group.finish();
}
