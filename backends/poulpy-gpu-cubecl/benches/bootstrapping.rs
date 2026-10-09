//! CKKS bootstrapping presets on the wgpu runtime: the portable client
//! generates the keys and encrypts on the host, the device backend compiles
//! the context, prepares the keys and bootstraps. The timed region is the
//! bootstrap, synchronized.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::suites::bench_ckks_bootstrapping_with_client;
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_gpu_cubecl::NTT4x30Wgpu;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_ckks_bootstrapping_with_client::<NTT4x30Wgpu, NTT4x30Portable>
}
criterion_main!(benches);
