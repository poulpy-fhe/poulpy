///! RUSTFLAGS="-C target-feature=+avx2,+fma" \
///!  cargo bench -p poulpy-bench --bench ckks_linear_transformation \
///!  --features ckks-bench,enable-avx
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "enable-avx")]
fn bench_ckks_linear_transformation(c: &mut Criterion) {
    poulpy_bench::bench_suite::ckks::bench_ckks_linear_transformation::<poulpy_cpu_avx::NTT120Avx>(c, "ntt120-avx");
}

#[cfg(not(feature = "enable-avx"))]
fn bench_ckks_linear_transformation(_c: &mut Criterion) {
    panic!("ckks_linear_transformation now benchmarks only cpu-avx; rerun with --features ckks-bench,enable-avx");
}

criterion_group! {
    name = benches;
    config = poulpy_bench::ckks_criterion_config();
    targets = bench_ckks_linear_transformation
}
criterion_main!(benches);
