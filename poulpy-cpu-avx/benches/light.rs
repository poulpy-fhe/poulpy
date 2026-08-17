//! Light benchmark binary: the same representative cross-section and
//! backend split as `standard.rs`, but the NTT-backend sweep is a single
//! size (`log_n` = 14) for a fast, CI-sized run. See
//! `poulpy_bench::{hal,core,schemes}::suites::light` for the shared
//! implementation; this binary just wires it up against `poulpy-cpu-avx`'s
//! backends.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::light::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::light::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::bench_binfhe;
use poulpy_bench::schemes::suites::light::bench_ckks;
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
     bench_binfhe::<Fft, CGGI>
}

criterion_main!(benches);
