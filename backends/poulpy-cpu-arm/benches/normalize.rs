//! Same-base and cross-base normalization at full and partial precision.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_arm::{FFT64Neon, NTT4x30Neon};

fn parallel(_c: &mut criterion::Criterion) {
    #[cfg(feature = "enable-rayon")]
    {
        bench_normalization::<poulpy_cpu_arm::FFT64NeonRayon>(_c);
        bench_normalization::<poulpy_cpu_arm::NTT4x30NeonRayon>(_c);
    }
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_normalization::<FFT64Neon>, bench_normalization::<NTT4x30Neon>, parallel
}
criterion_main!(benches);
