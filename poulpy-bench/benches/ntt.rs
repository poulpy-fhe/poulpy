use criterion::{Criterion, criterion_group, criterion_main};

pub fn bench_ntt_ref(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTable, Primes30};
    use std::hint::black_box;

    let group_name: String = "ntt_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTable<Primes30> = NttTable::<Primes30>::new(n);
        move || {
            NTT120Ref::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_intt_ref(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTableInv, Primes30};
    use std::hint::black_box;

    let group_name: String = "intt_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTableInv<Primes30> = NttTableInv::<Primes30>::new(n);
        move || {
            NTT120Ref::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_ntt_avx(_c: &mut Criterion) {
    eprintln!("Skipping: AVX NTT benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_ntt_avx(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::NTT120Avx;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTable, Primes30};
    use std::hint::black_box;

    let group_name: String = "ntt_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTable<Primes30> = NttTable::<Primes30>::new(n);
        move || {
            NTT120Avx::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
)))]
fn bench_ntt_ifma(_c: &mut Criterion) {
    eprintln!("Skipping: IFMA NTT benchmark requires x86_64 + AVX512F + AVX512IFMA + AVX512VL");
}

#[cfg(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
))]
pub fn bench_ntt_ifma(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx512::NTT126Ifma;
    use poulpy_cpu_avx512::ntt126_ifma_api::{Ntt126IfmaDFTExecute, Ntt126IfmaTable, Primes42};
    use std::hint::black_box;

    let group_name: String = "ntt_ifma_avx512".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: Ntt126IfmaTable<Primes42> = Ntt126IfmaTable::<Primes42>::new(n);
        move || {
            NTT126Ifma::ntt126_ifma_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
)))]
fn bench_intt_ifma(_c: &mut Criterion) {
    eprintln!("Skipping: IFMA INTT benchmark requires x86_64 + AVX512F + AVX512IFMA + AVX512VL");
}

#[cfg(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
))]
pub fn bench_intt_ifma(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx512::NTT126Ifma;
    use poulpy_cpu_avx512::ntt126_ifma_api::{Ntt126IfmaDFTExecute, Ntt126IfmaTableInv, Primes42};
    use std::hint::black_box;

    let group_name: String = "intt_ifma_avx512".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: Ntt126IfmaTableInv<Primes42> = Ntt126IfmaTableInv::<Primes42>::new(n);
        move || {
            NTT126Ifma::ntt126_ifma_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_intt_avx(_c: &mut Criterion) {
    eprintln!("Skipping: AVX INTT benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_intt_avx(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::NTT120Avx;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTableInv, Primes30};
    use std::hint::black_box;

    let group_name: String = "intt_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTableInv<Primes30> = NttTableInv::<Primes30>::new(n);
        move || {
            NTT120Avx::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f")))]
fn bench_ntt_avx512(_c: &mut Criterion) {
    eprintln!("Skipping: AVX-512 NTT benchmark requires x86_64 + AVX512F");
}

#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
pub fn bench_ntt_avx512(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx512::NTT120Avx512;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTable, Primes30};
    use std::hint::black_box;

    let group_name: String = "ntt_avx512f".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTable<Primes30> = NttTable::<Primes30>::new(n);
        move || {
            NTT120Avx512::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f")))]
fn bench_intt_avx512(_c: &mut Criterion) {
    eprintln!("Skipping: AVX-512 INTT benchmark requires x86_64 + AVX512F");
}

#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
pub fn bench_intt_avx512(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx512::NTT120Avx512;
    use poulpy_cpu_ref::reference::ntt120::{NttDFTExecute, NttTableInv, Primes30};
    use std::hint::black_box;

    let group_name: String = "intt_avx512f".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4 * n];
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);
        let table: NttTableInv<Primes30> = NttTableInv::<Primes30>::new(n);
        move || {
            NTT120Avx512::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_ntt_ref,
    bench_intt_ref,
    bench_ntt_avx,
    bench_intt_avx,
    bench_ntt_avx512,
    bench_intt_avx512,
    bench_ntt_ifma,
    bench_intt_ifma
}
criterion_main!(benches);
