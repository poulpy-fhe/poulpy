use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_bin_fhe::blind_rotation::CGGI;

fn bench_blind_rotate(c: &mut Criterion) {
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_ref::FFT64Ref, CGGI>(c, "fft64-ref");
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_avx::FFT64Avx, CGGI>(c, "fft64-avx");
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_avx512::FFT64Avx512, CGGI>(
        c,
        "fft64-avx512",
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_blind_rotate
}
criterion_main!(benches);
