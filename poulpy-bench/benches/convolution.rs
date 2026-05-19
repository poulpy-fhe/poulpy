use criterion::{Criterion, criterion_group, criterion_main};

fn bench_cnv_prepare_left(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_prepare_left, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_prepare_right(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_prepare_right, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_apply_dft, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_pairwise_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_pairwise_apply_dft, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_by_const_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_by_const_apply, &poulpy_bench::params::BenchParams::get().cnv; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_cnv_prepare_left,
    bench_cnv_prepare_right,
    bench_cnv_apply_dft,
    bench_cnv_pairwise_apply_dft,
    bench_cnv_by_const_apply
}
criterion_main!(benches);
