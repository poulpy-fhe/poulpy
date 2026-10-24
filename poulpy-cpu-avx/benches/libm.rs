//! `poulpy-libm` functions on the AVX2 NTT backend (`poulpy_bench::libm::suites`).
use criterion::{criterion_group, criterion_main};
use poulpy_bench::libm::suites::bench_libm;
use poulpy_cpu_avx::{FFT64AvxEncodingTable, NTT4x30Avx};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_libm::<NTT4x30Avx, FFT64AvxEncodingTable>
}
criterion_main!(benches);
