use criterion::{Criterion, criterion_group, criterion_main};

fn infos() -> poulpy_core::layouts::GLWELayout {
    let p = &poulpy_bench::params::BenchParams::get().core;
    poulpy_core::layouts::GLWELayout {
        n: p.n.into(),
        base2k: p.base2k.into(),
        k: p.k.into(),
        rank: p.rank.into(),
    }
}

fn bench_glwe_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_add_into, &infos(); c);
}
fn bench_glwe_add_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_add_assign, &infos(); c);
}
fn bench_glwe_sub(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_sub, &infos(); c);
}
fn bench_glwe_sub_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_sub_assign, &infos(); c);
}
fn bench_glwe_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_normalize, &infos(); c);
}
fn bench_glwe_normalize_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_normalize_assign, &infos(); c);
}
fn bench_glwe_mul_plain(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_mul_plain, &infos(); c);
}
fn bench_glwe_mul_plain_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::operations::bench_glwe_mul_plain_assign, &infos(); c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_add_into,
    bench_glwe_add_assign,
    bench_glwe_sub,
    bench_glwe_sub_assign,
    bench_glwe_normalize,
    bench_glwe_normalize_assign,
    bench_glwe_mul_plain,
    bench_glwe_mul_plain_assign
}
criterion_main!(benches);
