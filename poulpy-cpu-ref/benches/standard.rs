use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};

use poulpy_bench::params::{
    default_bench_params_ckks, default_bench_params_cnv, default_bench_params_core, default_bench_params_hal,
    default_bench_params_vmp,
};

type BE = poulpy_cpu_ref::NTT4x30Ref;

// ── Layer 1: HAL – full suite (every op family) ───────────────────────────────

fn hal(c: &mut Criterion) {
    use poulpy_bench::{
        bench_ops,
        hal::suites::{all_vec_znx_ops, convolution_ops, svp_ops, vmp_ops},
    };

    bench_ops(
        &all_vec_znx_ops::<BE, WallTime>(),
        default_bench_params_hal().as_slice(),
        "NTT4x30Ref",
        c,
    );
    bench_ops(
        &svp_ops::<BE, WallTime>(),
        default_bench_params_hal().as_slice(),
        "NTT4x30Ref",
        c,
    );
    bench_ops(
        &vmp_ops::<BE, WallTime>(),
        default_bench_params_vmp().as_slice(),
        "NTT4x30Ref",
        c,
    );
    bench_ops(
        &convolution_ops::<BE, WallTime>(),
        default_bench_params_cnv().as_slice(),
        "NTT4x30Ref",
        c,
    );
}

// ── Layer 2: Core – encryption ───────────────────────────────────────────────
fn core(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, core::suites::all_ops};

    bench_ops(
        &all_ops::<BE, WallTime>(),
        default_bench_params_core().as_slice(),
        "NTT4x30Ref",
        c,
    );
}

// ── Layer 3: Scheme ──────────────────────────────────────────────────────────

fn ckks(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, schemes::suites::all_ops};

    bench_ops(
        &all_ops::<BE, WallTime>(),
        default_bench_params_ckks().as_slice(),
        "NTT4x30Ref",
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
    ckks
}

criterion_main!(benches);
