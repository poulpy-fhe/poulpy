use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_add_into; c);
}
fn bench_vec_znx_add_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_add_assign; c);
}
fn bench_vec_znx_sub(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub; c);
}
fn bench_vec_znx_sub_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub_assign; c);
}
fn bench_vec_znx_sub_negate_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub_negate_assign; c);
}
fn bench_vec_znx_negate(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_negate; c);
}
fn bench_vec_znx_negate_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_negate_assign; c);
}
fn bench_vec_znx_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_normalize; c);
}
fn bench_vec_znx_normalize_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_normalize_assign; c);
}
fn bench_vec_znx_rotate(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rotate; c);
}
fn bench_vec_znx_rotate_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rotate_assign; c);
}
fn bench_vec_znx_automorphism(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_automorphism; c);
}
fn bench_vec_znx_automorphism_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_automorphism_assign; c);
}
fn bench_vec_znx_lsh(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_lsh; c);
}
fn bench_vec_znx_lsh_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_lsh_assign; c);
}
fn bench_vec_znx_rsh(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rsh; c);
}
fn bench_vec_znx_rsh_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rsh_assign; c);
}
fn bench_vec_znx_mul_xp_minus_one(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_mul_xp_minus_one; c);
}
fn bench_vec_znx_mul_xp_minus_one_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_mul_xp_minus_one_assign; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vec_znx_add_into,
    bench_vec_znx_add_assign,
    bench_vec_znx_sub,
    bench_vec_znx_sub_assign,
    bench_vec_znx_sub_negate_assign,
    bench_vec_znx_negate,
    bench_vec_znx_negate_assign,
    bench_vec_znx_normalize,
    bench_vec_znx_normalize_assign,
    bench_vec_znx_rotate,
    bench_vec_znx_rotate_assign,
    bench_vec_znx_automorphism,
    bench_vec_znx_automorphism_assign,
    bench_vec_znx_lsh,
    bench_vec_znx_lsh_assign,
    bench_vec_znx_rsh,
    bench_vec_znx_rsh_assign,
    bench_vec_znx_mul_xp_minus_one,
    bench_vec_znx_mul_xp_minus_one_assign
}
criterion_main!(benches);
