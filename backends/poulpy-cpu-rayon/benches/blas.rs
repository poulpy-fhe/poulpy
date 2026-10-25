//! BLAS layer on the portable Rayon NTT backend (`poulpy_bench::blas::suites`).
use criterion::{criterion_group, criterion_main};
use poulpy_bench::blas::suites::bench_blas;
use poulpy_cpu_rayon::NTT4x30PortableRayon;

criterion_group! {
    name = benches;
    config = poulpy_bench::blas_criterion_config();
    targets = bench_blas::<NTT4x30PortableRayon>
}
criterion_main!(benches);
