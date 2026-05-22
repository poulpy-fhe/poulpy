use criterion::{Criterion, criterion_group, criterion_main};

fn bench_coeff_matmul(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::coeff_mat::bench_coeff_matmul,
        &poulpy_bench::params::BenchParams::get().coeff_mat;
        c
    );
}

fn bench_coeff_matmul_bodies(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::coeff_mat::bench_coeff_matmul_bodies,
        &poulpy_bench::params::BenchParams::get().coeff_mat;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_coeff_matmul, bench_coeff_matmul_bodies
}
criterion_main!(benches);
