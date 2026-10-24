//! `poulpy-libm` functions on the AVX-512 NTT backends (`poulpy_bench::libm::suites`),
//! IFMA included with `enable-ifma`.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::libm::suites::bench_libm;
#[cfg(feature = "enable-ifma")]
use poulpy_cpu_avx512::NTT3x42Ifma;
use poulpy_cpu_avx512::{FFT64Avx512EncodingTable, NTT4x30Avx512};

#[cfg(not(feature = "enable-ifma"))]
criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_libm::<NTT4x30Avx512, FFT64Avx512EncodingTable>
}
#[cfg(feature = "enable-ifma")]
criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_libm::<NTT4x30Avx512, FFT64Avx512EncodingTable>, bench_libm::<NTT3x42Ifma, FFT64Avx512EncodingTable>
}
criterion_main!(benches);
