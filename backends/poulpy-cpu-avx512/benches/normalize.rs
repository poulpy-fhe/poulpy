//! Same-base and cross-base normalization at full and partial precision.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::hal::suites::bench_normalization;
use poulpy_cpu_avx512::{FFT64Avx512, NTT4x30Avx512};

fn parallel(_c: &mut criterion::Criterion) {
    #[cfg(feature = "enable-rayon")]
    {
        bench_normalization::<poulpy_cpu_avx512::FFT64Avx512Rayon>(_c);
        bench_normalization::<poulpy_cpu_avx512::NTT4x30Avx512Rayon>(_c);
    }
}

fn ifma(_c: &mut criterion::Criterion) {
    #[cfg(feature = "enable-ifma")]
    {
        bench_normalization::<poulpy_cpu_avx512::NTT3x42Ifma>(_c);
        #[cfg(feature = "enable-rayon")]
        bench_normalization::<poulpy_cpu_avx512::NTT3x42IfmaRayon>(_c);
    }
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_normalization::<FFT64Avx512>, bench_normalization::<NTT4x30Avx512>, parallel, ifma
}
criterion_main!(benches);
