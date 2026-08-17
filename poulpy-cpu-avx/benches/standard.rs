//! Standard benchmark binary: a small, representative cross-section of ops
//! (`hal::suites::standard_ops`, `core::suites::standard_ops`,
//! `schemes::suites::{ckks_standard_ops, bin_fhe_standard_ops}`), swept over
//! `log_n` in {13, 14, 15}.
//!
//! Scheme-level benchmarks run against the backend each scheme is meant for:
//! CKKS against the NTT backend, bin-fhe (blind rotation / circuit
//! bootstrapping) against the FFT backend. HAL and core — the shared
//! building blocks underneath both — run against both backends, each swept
//! at the sizes matching that backend's scheme: `log_n` 13/14/15 (matching
//! CKKS) for the NTT backend, and the bin-fhe ring degree for the FFT one.
//!
//! See `full.rs` for every op family over the full parameter grid, and
//! `light.rs` for this same cross-section at a single size.

use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};
use poulpy_bench::{
    bench_ops,
    core::params::default_bench_params_core,
    hal::params::{default_bench_params_hal, default_bench_params_vmp},
    schemes::params::{default_bench_params_blind_rotate, default_bench_params_circuit_bootstrapping, default_bench_params_ckks},
};
use poulpy_bin_fhe::blind_rotation::CGGI;

use poulpy_cpu_avx::FFT64Avx as Fft;
use poulpy_cpu_avx::NTT4x30Avx as Ntt;

const STANDARD_N: [u64; 3] = [1 << 13, 1 << 14, 1 << 15];

fn is_standard_n(n: u64) -> bool {
    STANDARD_N.contains(&n)
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
        default_bench_params_hal().into_iter().filter(|p| is_standard_n(p.n as u64)),
        c,
    );
    bench_ops(
        Ntt,
        &vmp_ops_ntt,
        default_bench_params_vmp().into_iter().filter(|p| is_standard_n(p.n as u64)),
        c,
    );

    let (hal_ops_fft, vmp_ops_fft) = standard_ops::<Fft, WallTime>();
    bench_ops(
        Fft {},
        &hal_ops_fft,
        default_bench_params_hal().into_iter().filter(|p| p.n as u64 == bin_fhe_n()),
        c,
    );
    bench_ops(
        Fft {},
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
        default_bench_params_core().into_iter().filter(|p| is_standard_n(p.n as u64)),
        c,
    );

    bench_ops(
        Fft {},
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
        default_bench_params_ckks().into_iter().filter(|p| is_standard_n(p.n as u64)),
        c,
    );
}

fn bin_fhe(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::bin_fhe_standard_ops;

    let (blind_rotate_ops, circuit_bootstrapping_ops) = bin_fhe_standard_ops::<Fft, CGGI, WallTime>();
    bench_ops(Fft {}, &blind_rotate_ops, [default_bench_params_blind_rotate()], c);
    bench_ops(
        Fft {},
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
