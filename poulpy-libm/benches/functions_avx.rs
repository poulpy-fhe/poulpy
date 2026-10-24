//! AVX2 function benchmarks.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx::{FFT64AvxReimTable, NTT4x30Avx};

fn functions(c: &mut Criterion) {
    common::bench_functions::<NTT4x30Avx, FFT64AvxReimTable>(c, "avx");
}

criterion_group!(benches, functions);
criterion_main!(benches);
