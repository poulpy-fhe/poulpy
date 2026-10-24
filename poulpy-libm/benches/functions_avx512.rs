//! AVX-512 function benchmarks.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx512::{FFT64Avx512ReimTable, NTT4x30Avx512};

fn functions(c: &mut Criterion) {
    common::bench_functions::<NTT4x30Avx512, FFT64Avx512ReimTable>(c, "avx512");
}

criterion_group!(benches, functions);
criterion_main!(benches);
