use criterion::{Criterion, criterion_group, criterion_main};

fn bench_ckks_sub(c: &mut Criterion) {
    poulpy_bench::for_each_ntt_backend!(poulpy_bench::bench_suite::ckks::bench_ckks_sub; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::ckks_criterion_config();
    targets = bench_ckks_sub
}
criterion_main!(benches);
