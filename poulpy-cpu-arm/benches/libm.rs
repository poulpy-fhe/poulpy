//! `poulpy-libm` functions on the NEON NTT backend (`poulpy_bench::libm::suites`).
use criterion::{criterion_group, criterion_main};
use poulpy_bench::libm::suites::bench_libm;
use poulpy_cpu_arm::{FFT64NeonEncodingTable, NTT4x30Neon};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_libm::<NTT4x30Neon, FFT64NeonEncodingTable>
}
criterion_main!(benches);
