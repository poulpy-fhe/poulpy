//! Full benchmark binary: every op family across HAL, core, CKKS, and
//! bin-fhe, swept over each layer's full default parameter grid (`log_n`
//! 10–15 for HAL, 12–16 for core/CKKS). This is the heaviest of the three
//! binaries — see `standard.rs` for a small representative cross-section at
//! a few sizes, and `light.rs` for a single-size, CI-sized smoke test.
//!
//! Scheme-level benchmarks run against the backend each scheme is meant for:
//! CKKS against the NTT backend, bin-fhe (blind rotation / circuit
//! bootstrapping) against the FFT backend. HAL and core — the shared
//! building blocks underneath both — run against both backends, each over
//! its full grid.

use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};

use poulpy_bench::bench_ops;
use poulpy_bench::core::params::default_bench_params_core;
use poulpy_bench::hal::params::{default_bench_params_cnv, default_bench_params_hal, default_bench_params_vmp};
use poulpy_bench::schemes::params::{
    default_bench_params_blind_rotate, default_bench_params_circuit_bootstrapping, default_bench_params_ckks,
};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_avx512::FFT64Avx512 as Fft;
use poulpy_cpu_avx512::NTT4x30Avx512 as Ntt;

// ── Layer 1: HAL – full suite (every op family) ───────────────────────────────

fn hal(c: &mut Criterion) {
    use poulpy_bench::hal::suites::{all_vec_znx_ops, convolution_ops, svp_ops, vmp_ops};

    bench_ops(Ntt, &all_vec_znx_ops::<Ntt, WallTime>(), default_bench_params_hal(), c);
    bench_ops(Ntt, &svp_ops::<Ntt, WallTime>(), default_bench_params_hal(), c);
    bench_ops(Ntt, &vmp_ops::<Ntt, WallTime>(), default_bench_params_vmp(), c);
    bench_ops(Ntt, &convolution_ops::<Ntt, WallTime>(), default_bench_params_cnv(), c);

    bench_ops(Fft, &all_vec_znx_ops::<Fft, WallTime>(), default_bench_params_hal(), c);
    bench_ops(Fft, &svp_ops::<Fft, WallTime>(), default_bench_params_hal(), c);
    bench_ops(Fft, &vmp_ops::<Fft, WallTime>(), default_bench_params_vmp(), c);
    bench_ops(Fft, &convolution_ops::<Fft, WallTime>(), default_bench_params_cnv(), c);
}

// ── Layer 2: Core – encryption ───────────────────────────────────────────────
fn core(c: &mut Criterion) {
    use poulpy_bench::core::suites::all_ops;

    bench_ops(Ntt, &all_ops::<Ntt, WallTime>(), default_bench_params_core(), c);
    bench_ops(Fft, &all_ops::<Fft, WallTime>(), default_bench_params_core(), c);
}

// ── Layer 3: Scheme ──────────────────────────────────────────────────────────

fn ckks(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::all_ops;

    bench_ops(Ntt, &all_ops::<Ntt, WallTime>(), default_bench_params_ckks(), c);
}

fn bin_fhe(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::bin_fhe_standard_ops;

    let (blind_rotate_ops, circuit_bootstrapping_ops) = bin_fhe_standard_ops::<Fft, CGGI, WallTime>();
    bench_ops(Fft, &blind_rotate_ops, [default_bench_params_blind_rotate()], c);
    bench_ops(
        Fft,
        &circuit_bootstrapping_ops,
        [default_bench_params_circuit_bootstrapping()],
        c,
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
    // Layer 1 – HAL FFT-domain,
    hal,
    // // Layer 2 – Core,
    core,
    // Layer 3 – Scheme,
    ckks,
    bin_fhe
}

criterion_main!(benches);
