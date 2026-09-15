//! Same-base and cross-base normalization at full and partial precision.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_normalization::<FFT64Portable>, bench_normalization::<NTT4x30Portable>
}
criterion_main!(benches);
