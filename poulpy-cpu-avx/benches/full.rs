//! Full benchmark binary: every op family across HAL, core, CKKS, and
//! bin-fhe, swept over each layer's full default parameter grid. See
//! `poulpy_bench::{hal,core,schemes}::suites` for the shared implementation;
//! this binary just wires it up against `poulpy-cpu-avx`'s backends.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::{bench_binfhe, bench_ckks, bench_ckks_bootstrapping};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_avx::FFT64Avx as Fft;
use poulpy_cpu_avx::NTT4x30Avx as Ntt;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_hal_ckks::<Ntt>,
     bench_hal_binfhe::<Fft>,
     bench_core_ckks::<Ntt>,
     bench_core_binfhe::<Fft>,
     bench_ckks::<Ntt>,
     bench_ckks_bootstrapping::<Ntt>,
     bench_ckks_bootstrapping::<Fft>,
     bench_binfhe::<Fft, CGGI>
}

criterion_main!(benches);
