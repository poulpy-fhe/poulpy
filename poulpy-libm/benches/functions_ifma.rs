//! AVX-512 IFMA function benchmarks.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx512::{FFT64Avx512ReimTable, NTT3x42Ifma};

fn functions(c: &mut Criterion) {
    common::bench_functions::<NTT3x42Ifma, FFT64Avx512ReimTable>(c, "ifma");
}

criterion_group!(benches, functions);
criterion_main!(benches);
