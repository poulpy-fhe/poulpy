use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_bin_fhe::blind_rotation::CGGI;

fn bench_circuit_bootstrapping(c: &mut Criterion) {
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_ref::FFT64Ref, CGGI>(
        c,
        "fft64-ref",
    );
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_avx::FFT64Avx, CGGI>(
        c,
        "fft64-avx",
    );
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_avx512::FFT64Avx512, CGGI>(
        c,
        "fft64-avx512",
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_circuit_bootstrapping
}
criterion_main!(benches);
