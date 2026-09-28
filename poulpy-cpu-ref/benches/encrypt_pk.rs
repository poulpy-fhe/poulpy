//! Public-key encryption under a prepared key.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::bench_glwe_encrypt_pk;
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_encrypt_pk::<FFT64Ref>, bench_glwe_encrypt_pk::<NTT4x30Ref>
}
criterion_main!(benches);
