use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

fn glwe_infos() -> GLWELayout {
    let p = &poulpy_bench::params::BenchParams::get().core;
    GLWELayout {
        n: p.n.into(),
        base2k: p.base2k.into(),
        k: p.k.into(),
        rank: p.rank.into(),
    }
}

fn bench_glwe_tensor_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_apply, &glwe_infos(); c);
}

fn bench_glwe_tensor_apply_add_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_apply_add_assign,
        &glwe_infos();
        c
    );
}

fn bench_glwe_tensor_prepare_left(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_prepare_left,
        &glwe_infos();
        c
    );
}

fn bench_glwe_tensor_prepare_right(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_prepare_right,
        &glwe_infos();
        c
    );
}

fn bench_glwe_tensor_diag_lane(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_diag_lane,
        &glwe_infos();
        c
    );
}

fn bench_glwe_tensor_pairwise_lane(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_pairwise_lane,
        &glwe_infos();
        c
    );
}

fn bench_glwe_tensor_square_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::glwe_tensor::bench_glwe_tensor_square_apply,
        &glwe_infos();
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_tensor_apply,
    bench_glwe_tensor_apply_add_assign,
    bench_glwe_tensor_prepare_left,
    bench_glwe_tensor_prepare_right,
    bench_glwe_tensor_diag_lane,
    bench_glwe_tensor_pairwise_lane,
    bench_glwe_tensor_square_apply
}
criterion_main!(benches);
