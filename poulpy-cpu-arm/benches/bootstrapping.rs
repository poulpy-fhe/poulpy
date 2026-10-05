//! CKKS bootstrapping presets on the NTT backend, serial and Rayon.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::suites::bench_ckks_bootstrapping;
use poulpy_cpu_arm::NTT4x30Neon;

fn parallel(_c: &mut criterion::Criterion) {
    #[cfg(feature = "enable-rayon")]
    bench_ckks_bootstrapping::<poulpy_cpu_arm::NTT4x30NeonRayon, 52>(_c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_ckks_bootstrapping::<NTT4x30Neon, 52>, parallel
}
criterion_main!(benches);
