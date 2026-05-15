use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTExecute, ReimFFTRef, ReimFFTTable, ReimIFFTRef, ReimIFFTTable};

pub fn bench_fft_ref(c: &mut Criterion) {
    let group_name: String = "fft_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);
        move || {
            ReimFFTRef::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_ifft_ref(c: &mut Criterion) {
    let group_name: String = "ifft_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);
        move || {
            ReimIFFTRef::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
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
fn bench_fft_avx(_c: &mut Criterion) {
    eprintln!("Skipping: AVX FFT benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_fft_avx(c: &mut Criterion) {
    use poulpy_cpu_avx::ReimFFTAvx;

    let group_name: String = "fft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);
        move || {
            ReimFFTAvx::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
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
fn bench_ifft_avx(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFFT benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_ifft_avx(c: &mut Criterion) {
    use poulpy_cpu_avx::ReimIFFTAvx;

    let group_name: String = "ifft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);
        move || {
            ReimIFFTAvx::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(feature = "enable-neon", target_arch = "aarch64")))]
fn bench_fft_neon(_c: &mut Criterion) {
    eprintln!("Skipping: NEON FFT benchmark requires aarch64 + enable-neon");
}

#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
pub fn bench_fft_neon(c: &mut Criterion) {
    use poulpy_cpu_arm::FFT64Neon;
    use poulpy_cpu_ref::reference::fft64::reim::ReimFFTExecute;

    let group_name: String = "fft_neon".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);
        move || {
            <FFT64Neon as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(not(all(feature = "enable-neon", target_arch = "aarch64")))]
fn bench_ifft_neon(_c: &mut Criterion) {
    eprintln!("Skipping: NEON IFFT benchmark requires aarch64 + enable-neon");
}

#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
pub fn bench_ifft_neon(c: &mut Criterion) {
    use poulpy_cpu_arm::FFT64Neon;
    use poulpy_cpu_ref::reference::fft64::reim::ReimFFTExecute;

    let group_name: String = "ifft_neon".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale = 1.0f64 / (2 * m) as f64;
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);
        move || {
            <FFT64Neon as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_fft_ref,
    bench_ifft_ref,
    bench_fft_avx,
    bench_ifft_avx,
    bench_fft_neon,
    bench_ifft_neon
}
criterion_main!(benches);
