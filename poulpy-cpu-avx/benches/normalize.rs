//! Normalization sweeps for reference and AVX2 FFT64 and NTT4x30 backends.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_avx::{FFT64Avx, NTT4x30Avx};
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
        bench_normalization::<FFT64Ref>,
        bench_normalization::<NTT4x30Ref>,
        bench_normalization::<FFT64Avx>,
        bench_normalization::<NTT4x30Avx>
}

criterion_main!(benches);
