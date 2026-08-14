use criterion::{Criterion, criterion_group, criterion_main};

fn bench_cnv_prepare_left_pvec(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_prepare_left_pvec, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_prepare_right_pvec(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_prepare_right_pvec, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_apply_pvec_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_apply_pvec_to_dft, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_apply_pvec_to_dft_accumulate(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_apply_pvec_to_dft_accumulate, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_pairwise_apply_pvec_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_pairwise_apply_pvec_to_dft, &poulpy_bench::params::BenchParams::get().cnv; c);
}
fn bench_cnv_by_const_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::convolution::bench_cnv_by_const_apply, &poulpy_bench::params::BenchParams::get().cnv; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_cnv_prepare_left_pvec,
    bench_cnv_prepare_right_pvec,
    bench_cnv_apply_pvec_to_dft,
    bench_cnv_apply_pvec_to_dft_accumulate,
    bench_cnv_pairwise_apply_pvec_to_dft,
    bench_cnv_by_const_apply
}
criterion_main!(benches);
