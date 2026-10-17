//! Full benchmark binary: every op family across HAL, core, CKKS, and
//! bin-fhe, swept over each layer's full default parameter grid. See
//! `poulpy_bench::{hal,core,schemes}::suites` for the shared implementation;
//! this binary just wires it up against `poulpy-cpu-avx512`'s backends.
//!
//! With `enable-blas`, a second group runs the BLAS layer on the NTT backend
//! (`poulpy_bench::blas::suites`) under its own Criterion configuration.

use criterion::{Criterion, criterion_group};
#[cfg(feature = "enable-blas")]
use poulpy_bench::blas::suites::bench_blas;
use poulpy_bench::core::suites::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::{bench_binfhe, bench_ckks, bench_ckks_bootstrapping};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_avx512::FFT64Avx512 as Fft;
#[cfg(feature = "enable-ifma")]
use poulpy_cpu_avx512::NTT3x42Ifma as Ifma;
use poulpy_cpu_avx512::NTT4x30Avx512 as Ntt;

fn bench_ckks_bootstrapping_ifma(_c: &mut Criterion) {
    #[cfg(feature = "enable-ifma")]
    bench_ckks_bootstrapping::<Ifma, 52>(_c);
}

fn bench_ckks_bootstrapping_ifma_rayon(_c: &mut Criterion) {
    #[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
    bench_ckks_bootstrapping::<poulpy_cpu_avx512::NTT3x42IfmaRayon, 52>(_c);
}

/// HAL sweep for `NTT3x42Ifma`, the one accelerated backend that had no HAL row.
fn bench_hal_ifma(_c: &mut Criterion) {
    #[cfg(feature = "enable-ifma")]
    {
        bench_hal_ckks::<Ifma>(_c);
        bench_hal_binfhe::<Ifma>(_c);
    }
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_hal_ckks::<Ntt>,
     bench_hal_binfhe::<Fft>,
     bench_core_ckks::<Ntt>,
     bench_core_binfhe::<Fft>,
     bench_ckks::<Ntt>,
     bench_ckks_bootstrapping::<Ntt, 52>,
     bench_hal_ifma,
     bench_ckks_bootstrapping_ifma,
     bench_ckks_bootstrapping_ifma_rayon,
     bench_ckks_bootstrapping::<Fft, 18>,
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
