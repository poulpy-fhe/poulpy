use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::ckks_encoding::bench_encoding;
use poulpy_cpu_avx512::FFT64Avx512;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_encoding::<FFT64Avx512, f64>, bench_encoding::<FFT64Avx512, poulpy_ckks::Quad>
}
criterion_main!(benches);
