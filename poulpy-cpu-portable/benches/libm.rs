//! `poulpy-libm` functions on the portable NTT backend (`poulpy_bench::libm::suites`).
use criterion::{criterion_group, criterion_main};
use poulpy_bench::libm::suites::bench_libm;
use poulpy_cpu_portable::{NTT4x30Portable, ckks_encoding::EncodingFFTTable};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_libm::<NTT4x30Portable, EncodingFFTTable<f64>>
}
criterion_main!(benches);
