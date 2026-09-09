//! Normalization sweeps for reference and AVX2 FFT64 and NTT4x30 backends.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_avx::{FFT64Avx, NTT4x30Avx};
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};

fn parallel(_c: &mut criterion::Criterion) {
    #[cfg(feature = "enable-rayon")]
    {
        bench_normalization::<poulpy_cpu_avx::FFT64AvxRayon>(_c);
        bench_normalization::<poulpy_cpu_avx::NTT4x30AvxRayon>(_c);
    }
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
        bench_normalization::<FFT64Ref>,
        bench_normalization::<NTT4x30Ref>,
        bench_normalization::<FFT64Avx>,
        bench_normalization::<NTT4x30Avx>,
        parallel
}

criterion_main!(benches);
