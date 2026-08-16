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

use poulpy_bench::core::params::default_bench_params_core;
use poulpy_bench::hal::params::{default_bench_params_cnv, default_bench_params_hal, default_bench_params_vmp};
use poulpy_bench::schemes::params::{
    default_bench_params_blind_rotate, default_bench_params_circuit_bootstrapping, default_bench_params_ckks,
};
use poulpy_bin_fhe::blind_rotation::CGGI;

type NTT = poulpy_cpu_ref::NTT4x30Ref;
type FFT = poulpy_cpu_ref::FFT64Ref;

// ── Layer 1: HAL – full suite (every op family) ───────────────────────────────

fn hal(c: &mut Criterion) {
    use poulpy_bench::{
        bench_ops,
        hal::suites::{all_vec_znx_ops, convolution_ops, svp_ops, vmp_ops},
    };

    bench_ops(
        &all_vec_znx_ops::<NTT, WallTime>(),
        default_bench_params_hal().as_slice(),
        "NTT4x30Ref/hal",
        c,
    );
    bench_ops(
        &svp_ops::<NTT, WallTime>(),
        default_bench_params_hal().as_slice(),
        "NTT4x30Ref/hal",
        c,
    );
    bench_ops(
        &vmp_ops::<NTT, WallTime>(),
        default_bench_params_vmp().as_slice(),
        "NTT4x30Ref/hal",
        c,
    );
    bench_ops(
        &convolution_ops::<NTT, WallTime>(),
        default_bench_params_cnv().as_slice(),
        "NTT4x30Ref/hal",
        c,
    );

    bench_ops(
        &all_vec_znx_ops::<FFT, WallTime>(),
        default_bench_params_hal().as_slice(),
        "FFT64Ref/hal",
        c,
    );
    bench_ops(
        &svp_ops::<FFT, WallTime>(),
        default_bench_params_hal().as_slice(),
        "FFT64Ref/hal",
        c,
    );
    bench_ops(
        &vmp_ops::<FFT, WallTime>(),
        default_bench_params_vmp().as_slice(),
        "FFT64Ref/hal",
        c,
    );
    bench_ops(
        &convolution_ops::<FFT, WallTime>(),
        default_bench_params_cnv().as_slice(),
        "FFT64Ref/hal",
        c,
    );
}

// ── Layer 2: Core – encryption ───────────────────────────────────────────────
fn core(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, core::suites::all_ops};

    bench_ops(
        &all_ops::<NTT, WallTime>(),
        default_bench_params_core().as_slice(),
        "NTT4x30Ref/core",
        c,
    );
    bench_ops(
        &all_ops::<FFT, WallTime>(),
        default_bench_params_core().as_slice(),
        "FFT64Ref/core",
        c,
    );
}

// ── Layer 3: Scheme ──────────────────────────────────────────────────────────

fn ckks(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, schemes::suites::all_ops};

    bench_ops(
        &all_ops::<NTT, WallTime>(),
        default_bench_params_ckks().as_slice(),
        "NTT4x30Ref/ckks",
        c,
    );
}

fn bin_fhe(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, schemes::suites::bin_fhe_standard_ops};

    let (blind_rotate_ops, circuit_bootstrapping_ops) = bin_fhe_standard_ops::<FFT, CGGI, WallTime>();
    bench_ops(&blind_rotate_ops, &[default_bench_params_blind_rotate()], "FFT64Ref/bin_fhe", c);
    bench_ops(
        &circuit_bootstrapping_ops,
        &[default_bench_params_circuit_bootstrapping()],
        "FFT64Ref/bin_fhe",
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
