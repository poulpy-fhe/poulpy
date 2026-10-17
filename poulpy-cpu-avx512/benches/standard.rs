//! Standard benchmark binary: a small, representative cross-section of ops,
//! swept over `log_n` in {13, 14, 15}. See
//! `poulpy_bench::{hal,core,schemes}::suites::standard` for the shared
//! implementation; this binary just wires it up against
//! `poulpy-cpu-avx512`'s backends.
//!
//! See `full.rs` for every op family over the full parameter grid, and
//! `light.rs` for this same cross-section at a single size.
//!
//! With `enable-blas`, a second group runs the BLAS layer on the NTT backend
//! (`poulpy_bench::blas::suites`) under its own Criterion configuration.

use criterion::criterion_group;
#[cfg(feature = "enable-blas")]
use poulpy_bench::blas::suites::standard::bench_blas;
use poulpy_bench::core::suites::standard::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::standard::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::bench_binfhe;
use poulpy_bench::schemes::suites::standard::bench_ckks;
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_avx512::FFT64Avx512 as Fft;
use poulpy_cpu_avx512::NTT4x30Avx512 as Ntt;

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

// What `criterion_main!(benches)` expands to, with the BLAS group between the others and the final summary.
fn main() {
    benches();
    #[cfg(feature = "enable-blas")]
    blas_benches();

    criterion::Criterion::default().configure_from_args().final_summary();
}
