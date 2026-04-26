use criterion::{Criterion, criterion_group, criterion_main};

fn bench_svp_prepare(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::svp::bench_svp_prepare, &poulpy_bench::params::BenchParams::get().svp_prepare; c);
}
fn bench_svp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::svp::bench_svp_apply_dft_to_dft, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_svp_apply_dft_to_dft_assign(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_bench::bench_suite::hal::svp::bench_svp_apply_dft_to_dft_assign, &poulpy_bench::params::BenchParams::get().hal; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_svp_prepare,
    bench_svp_apply_dft_to_dft,
    bench_svp_apply_dft_to_dft_assign
}
criterion_main!(benches);
