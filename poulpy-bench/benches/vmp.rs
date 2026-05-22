use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vmp_prepare(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_prepare, &poulpy_bench::params::BenchParams::get().vmp; c);
}
fn bench_vmp_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_apply_dft, &poulpy_bench::params::BenchParams::get().vmp; c);
}
fn bench_vmp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_apply_dft_to_dft, &poulpy_bench::params::BenchParams::get().vmp; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vmp_prepare,
    bench_vmp_apply_dft,
    bench_vmp_apply_dft_to_dft
}
criterion_main!(benches);
