//! Portable function benchmarks.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_portable::{NTT4x30Portable, ckks_encoding::EncodingFFTTable};

fn functions(c: &mut Criterion) {
    common::bench_functions::<NTT4x30Portable, EncodingFFTTable<f64>>(c, "portable");
}

criterion_group!(benches, functions);
criterion_main!(benches);
