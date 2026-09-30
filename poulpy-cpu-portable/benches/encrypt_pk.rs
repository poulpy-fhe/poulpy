//! Public-key encryption under a prepared key.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::bench_glwe_encrypt_pk;
use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_encrypt_pk::<FFT64Portable>, bench_glwe_encrypt_pk::<NTT4x30Portable>
}
criterion_main!(benches);
