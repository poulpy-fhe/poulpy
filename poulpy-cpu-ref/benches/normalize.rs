//! Same-base and cross-base normalization at full and partial precision.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_normalization::<FFT64Ref>, bench_normalization::<NTT4x30Ref>
}
criterion_main!(benches);
