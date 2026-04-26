use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_dft_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_add_into, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_add_assign(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_add_assign, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_apply, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_idft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_idft_apply, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_idft_apply_tmpa(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_idft_apply_tmpa, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub_assign(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub_assign, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub_negate_assign(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub_negate_assign, &poulpy_bench::params::BenchParams::get().hal; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vec_znx_dft_add_into,
    bench_vec_znx_dft_add_assign,
    bench_vec_znx_dft_apply,
    bench_vec_znx_idft_apply,
    bench_vec_znx_idft_apply_tmpa,
    bench_vec_znx_dft_sub,
    bench_vec_znx_dft_sub_assign,
    bench_vec_znx_dft_sub_negate_assign
}
criterion_main!(benches);
