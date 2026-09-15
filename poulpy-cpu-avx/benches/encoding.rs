use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::ckks_encoding::bench_encoding;
use poulpy_cpu_avx::FFT64Avx;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_encoding::<FFT64Avx, f64>, bench_encoding::<FFT64Avx, poulpy_ckks::Quad>
}
criterion_main!(benches);
