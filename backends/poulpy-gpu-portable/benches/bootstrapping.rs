//! CKKS bootstrapping presets on every runtime enabled by a feature: the
//! portable client generates the keys and encrypts on the host, the device
//! backend compiles the context, prepares the keys and bootstraps.
//!
//! Two groups per runtime. `ckks_bootstrapping` is the shared benchmark of
//! `poulpy-bench`: each iteration calls the bootstrap. `ckks_bootstrapping_replayed`
//! records the bootstrap once with [`poulpy_gpu_portable::Replay`] and times
//! its replays, the way an application that repeats a bootstrap on the same
//! buffers would run it. Every iteration ends with a device synchronization.

use criterion::{Criterion, criterion_group, criterion_main};

#[allow(unused_macros)]
macro_rules! bootstrapping {
    ($c:expr, $backend:ty, $runtime:ty, $name:literal) => {{
        use poulpy_ckks::{
            presets::bootstrapping::all,
            test_suite::presets::{BootstrappingPresetRun, preset_for_backend},
        };
        use poulpy_cpu_portable::NTT4x30Portable;
        use std::{cell::RefCell, rc::Rc};

        poulpy_bench::schemes::suites::bench_ckks_bootstrapping_with_client::<$backend, NTT4x30Portable>($c);

        // The name the shared benchmark gives the backend type.
        let backend = $name;
        let mut group = $c.benchmark_group(format!("{backend}/ckks/ckks_bootstrapping_replayed"));
        group.sample_size(10);
        for preset in all().unwrap() {
            let preset = preset_for_backend::<
                $backend,
                poulpy_gpu_portable::NTT4x30Gpu<$runtime, poulpy_hal::layouts::ConjugateInvariant>,
            >(&preset)
            .unwrap();
            let advertised = preset.log2_precision() as f64;
            let id = preset.name().to_string();
            // Set up on the first call of the closure, so that a filtered-out preset costs nothing.
            let mut recorded = None;
            group.bench_function(criterion::BenchmarkId::from_parameter(&id), |bencher| {
                let (run, replay) = recorded.get_or_insert_with(|| {
                    let run = Rc::new(RefCell::new(match preset.plan().eval_mod_ring() {
                        poulpy_ckks::layouts::EvalModRing::Standard => {
                            BootstrappingPresetRun::<$backend, NTT4x30Portable>::setup(preset.clone())
                        }
                        _ => BootstrappingPresetRun::<$backend, NTT4x30Portable>::setup_ci(preset.clone()),
                    }));
                    let body = run.clone();
                    let replay = poulpy_gpu_portable::Replay::<$runtime>::record(move || body.borrow_mut().bootstrap_queued());
                    (run, replay)
                });
                bencher.iter(|| {
                    replay.replay();
                    run.borrow().synchronize();
                });
            });
            if let Some((run, replay)) = recorded {
                println!(
                    "REPLAY backend={backend} preset={id} graph={} launches={} retained_bytes={}",
                    replay.is_graph(),
                    replay.launches(),
                    replay.retained_bytes()
                );
                drop(replay);
                let (re, im) = run.borrow_mut().precision();
                println!(
                    "PRECISION backend={backend} preset={id} re_min={:.2}b im_min={:.2}b advertised={advertised}b",
                    re.min_log2_prec, im.min_log2_prec
                );
                assert!(
                    re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
                    "preset {id} advertises {advertised} bits of precision"
                );
            }
        }
        group.finish();
    }};
}

#[allow(unused_variables)]
fn runtimes(c: &mut Criterion) {
    #[cfg(feature = "enable-cuda")]
    bootstrapping!(
        c,
        poulpy_gpu_portable::NTT4x30Cuda,
        poulpy_gpu_portable::Cuda,
        "NTT4x30Gpu<Cuda>"
    );
    #[cfg(feature = "enable-hip")]
    bootstrapping!(
        c,
        poulpy_gpu_portable::NTT4x30Hip,
        poulpy_gpu_portable::Hip,
        "NTT4x30Gpu<Hip>"
    );
    #[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
    bootstrapping!(
        c,
        poulpy_gpu_portable::NTT4x30Wgpu,
        poulpy_gpu_portable::Wgpu,
        "NTT4x30Gpu<Wgpu>"
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = runtimes
}
criterion_main!(benches);
