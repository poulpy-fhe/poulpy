//! Light benchmark binary: the same representative cross-section and
//! backend split as `standard.rs`, but the NTT-backend sweep is a single
//! size (`log_n` = 14) for a fast, CI-sized run. See
//! `poulpy_bench::{hal,core,schemes}::suites::light` for the shared
//! implementation; this binary just wires it up against `poulpy-cpu-arm`'s
//! backends.
//!
//! With `enable-blas`, a second group runs the BLAS layer on the NTT backend
//! (`poulpy_bench::blas::suites`) under its own Criterion configuration.

use criterion::{criterion_group, criterion_main};
#[cfg(feature = "enable-blas")]
use poulpy_bench::blas::suites::light::bench_blas;
use poulpy_bench::core::suites::light::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::light::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::bench_binfhe;
use poulpy_bench::schemes::suites::light::bench_ckks;
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_arm::FFT64Neon as Fft;
use poulpy_cpu_arm::NTT4x30Neon as Ntt;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_hal_ckks::<Ntt>,
     bench_hal_binfhe::<Fft>,
     bench_core_ckks::<Ntt>,
     bench_core_binfhe::<Fft>,
     bench_ckks::<Ntt>,
     bench_binfhe::<Fft, CGGI>
}

#[cfg(feature = "enable-blas")]
criterion_group! {
    name = blas_benches;
    config = poulpy_bench::blas_criterion_config();
    targets = bench_blas::<Ntt>
}

#[cfg(not(feature = "enable-blas"))]
criterion_main!(benches);
#[cfg(feature = "enable-blas")]
criterion_main!(benches, blas_benches);
