//! Light benchmark binary: the same representative cross-section and
//! backend split as `standard.rs`, but the NTT-backend sweep is a single
//! size (`log_n` = 14) for a fast, CI-sized run.

use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};
use poulpy_bench::{
    bench_ops,
    core::params::default_bench_params_core,
    hal::params::{default_bench_params_hal, default_bench_params_vmp},
    schemes::params::{default_bench_params_blind_rotate, default_bench_params_circuit_bootstrapping, default_bench_params_ckks},
};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_ref::FFT64Ref as Fft;
use poulpy_cpu_ref::NTT4x30Ref as Ntt;

const LIGHT_N: u64 = 1 << 14;

fn is_light_n(n: u64) -> bool {
    n == LIGHT_N
}

/// The single ring degree bin-fhe's representative params use — HAL/core's
/// FFT-backend run is swept at this one size to match.
fn bin_fhe_n() -> u64 {
    default_bench_params_blind_rotate().bin_fhe_params.n_glwe as u64
}

// ── Layer 1: HAL ────────────────────────────────────────────────────────────

fn hal(c: &mut Criterion) {
    use poulpy_bench::hal::suites::standard_ops;

    let (hal_ops_ntt, vmp_ops_ntt) = standard_ops::<Ntt, WallTime>();
    bench_ops(
        Ntt,
        &hal_ops_ntt,
        default_bench_params_hal().into_iter().filter(|p| is_light_n(p.n as u64)),
        c,
    );
    bench_ops(
        Ntt,
        &vmp_ops_ntt,
        default_bench_params_vmp().into_iter().filter(|p| is_light_n(p.n as u64)),
        c,
    );

    let (hal_ops_fft, vmp_ops_fft) = standard_ops::<Fft, WallTime>();
    bench_ops(
        Fft,
        &hal_ops_fft,
        default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
    bench_ops(
        Fft,
        &vmp_ops_fft,
        default_bench_params_vmp().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
}

// ── Layer 2: Core ───────────────────────────────────────────────────────────

fn core(c: &mut Criterion) {
    use poulpy_bench::core::suites::standard_ops;

    bench_ops(
        Ntt,
        &standard_ops::<Ntt, WallTime>(),
        default_bench_params_core().into_iter().filter(|p| is_light_n(p.n as u64)),
        c,
    );

    bench_ops(
        Fft,
        &standard_ops::<Fft, WallTime>(),
        default_bench_params_core().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
}

// ── Layer 3: Scheme ─────────────────────────────────────────────────────────

fn ckks(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::ckks_standard_ops;

    bench_ops(
        Ntt,
        &ckks_standard_ops::<Ntt, WallTime>(),
        default_bench_params_ckks().into_iter().filter(|p| is_light_n(p.n as u64)),
        c,
    );
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
    targets = hal, core, ckks, bin_fhe
}

criterion_main!(benches);
