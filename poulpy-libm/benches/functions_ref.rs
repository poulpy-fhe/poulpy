//! Portable function benchmarks.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::{FFT64ReimTable, NTT4x30Ref};

fn functions(c: &mut Criterion) {
    common::bench_functions::<NTT4x30Ref, FFT64ReimTable<f64>>(c, "ref");
}

criterion_group!(benches, functions);
criterion_main!(benches);
