//! Public-key encryption under a prepared key.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::bench_glwe_encrypt_pk;
use poulpy_cpu_avx512::{FFT64Avx512, NTT4x30Avx512};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_encrypt_pk::<FFT64Avx512>, bench_glwe_encrypt_pk::<NTT4x30Avx512>
}
criterion_main!(benches);
