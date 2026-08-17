//! Standard benchmark binary: a small, representative cross-section of ops,
//! swept over `log_n` in {13, 14, 15}. See
//! `poulpy_bench::{hal,core,schemes}::suites::standard` for the shared
//! implementation; this binary just wires it up against `poulpy-cpu-ref`'s
//! backends.
//!
//! See `full.rs` for every op family over the full parameter grid, and
//! `light.rs` for this same cross-section at a single size.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::standard::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::standard::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::bench_binfhe;
use poulpy_bench::schemes::suites::standard::bench_ckks;
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_ref::FFT64Ref as Fft;
use poulpy_cpu_ref::NTT4x30Ref as Ntt;

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
