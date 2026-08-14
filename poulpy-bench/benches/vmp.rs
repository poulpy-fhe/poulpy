use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vmp_prepare_pmat(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_prepare_pmat, &poulpy_bench::params::BenchParams::get().vmp; c);
}
fn bench_vmp_apply_pmat_small_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_apply_pmat_small_to_dft, &poulpy_bench::params::BenchParams::get().vmp; c);
}
fn bench_vmp_apply_pmat_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_apply_pmat_dft_to_dft, &poulpy_bench::params::BenchParams::get().vmp; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vmp_prepare_pmat,
    bench_vmp_apply_pmat_small_to_dft,
    bench_vmp_apply_pmat_dft_to_dft
}
criterion_main!(benches);
