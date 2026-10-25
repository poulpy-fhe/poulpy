//! Full benchmark binary: every op family across HAL, core, CKKS, and
//! bin-fhe, swept over each layer's full default parameter grid. See
//! `poulpy_bench::{hal,core,schemes}::suites` for the shared implementation;
//! this binary just wires it up against `poulpy-cpu-portable`'s backends.
//!
//! With `enable-blas`, a second group runs the BLAS layer on the NTT backend
//! (`poulpy_bench::blas::suites`) under its own Criterion configuration.

use criterion::{criterion_group, criterion_main};
#[cfg(feature = "enable-blas")]
use poulpy_bench::blas::suites::bench_blas;
use poulpy_bench::core::suites::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::libm::suites::bench_libm;
use poulpy_bench::schemes::suites::{bench_binfhe, bench_ckks, bench_ckks_bootstrapping};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_portable::FFT64Portable as Fft;
use poulpy_cpu_portable::NTT4x30Portable as Ntt;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_hal_ckks::<Ntt>,
     bench_hal_binfhe::<Fft>,
     bench_core_ckks::<Ntt>,
     bench_core_binfhe::<Fft>,
     bench_ckks::<Ntt>,
     bench_libm::<Ntt>,
     bench_ckks_bootstrapping::<Ntt>,
     bench_ckks_bootstrapping::<Fft>,
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
