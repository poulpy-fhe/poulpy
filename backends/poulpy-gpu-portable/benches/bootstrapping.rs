//! CKKS bootstrapping presets on every runtime enabled by a feature: the
//! portable client generates the keys and encrypts on the host, the device
//! backend compiles the context, prepares the keys and bootstraps.
//!
//! Two groups per runtime, from `poulpy-bench`: `ckks_bootstrapping` times
//! the bootstrap call, `ckks_bootstrapping_replayed` the replays of a
//! bootstrap recorded once with [`poulpy_gpu_portable::Replay`]. Every
//! iteration ends with a device synchronization.

use criterion::{Criterion, criterion_group, criterion_main};

#[allow(unused_macros)]
macro_rules! bootstrapping {
    ($c:expr, $backend:ty, $runtime:ty) => {{
        use poulpy_bench::schemes::suites::{
            bench_ckks_bootstrapping_replayed_with_client, bench_ckks_bootstrapping_with_client,
        };
        use poulpy_cpu_portable::NTT4x30Portable;

        bench_ckks_bootstrapping_with_client::<$backend, NTT4x30Portable>($c);
        bench_ckks_bootstrapping_replayed_with_client::<$backend, NTT4x30Portable>($c, |body| {
            let mut replay = poulpy_gpu_portable::Replay::<$runtime>::record(body);
            println!(
                "REPLAY graph={} launches={} retained_bytes={}",
                replay.is_graph(),
                replay.launches(),
                replay.retained_bytes()
            );
            Box::new(move || replay.replay())
        });
    }};
}

#[allow(unused_variables)]
fn runtimes(c: &mut Criterion) {
    #[cfg(feature = "enable-cuda")]
    bootstrapping!(c, poulpy_gpu_portable::NTT4x30Cuda, poulpy_gpu_portable::Cuda);
    #[cfg(feature = "enable-hip")]
    bootstrapping!(c, poulpy_gpu_portable::NTT4x30Hip, poulpy_gpu_portable::Hip);
    #[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
    bootstrapping!(c, poulpy_gpu_portable::NTT4x30Wgpu, poulpy_gpu_portable::Wgpu);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = runtimes
}
criterion_main!(benches);
