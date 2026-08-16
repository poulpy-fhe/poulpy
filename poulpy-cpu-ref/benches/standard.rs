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

type NTT = poulpy_cpu_ref::NTT4x30Ref;
type FFT = poulpy_cpu_ref::FFT64Ref;

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

    let (hal_ops_ntt, vmp_ops_ntt) = standard_ops::<NTT, WallTime>();
    let hal_params_ntt: Vec<_> = default_bench_params_hal()
        .into_iter()
        .filter(|p| is_standard_n(p.n as u64))
        .collect();
    let vmp_params_ntt: Vec<_> = default_bench_params_vmp()
        .into_iter()
        .filter(|p| is_standard_n(p.n as u64))
        .collect();
    bench_ops(&hal_ops_ntt, hal_params_ntt.as_slice(), "NTT4x30Ref/hal", c);
    bench_ops(&vmp_ops_ntt, vmp_params_ntt.as_slice(), "NTT4x30Ref/hal", c);

    let (hal_ops_fft, vmp_ops_fft) = standard_ops::<FFT, WallTime>();
    let hal_params_fft: Vec<_> = default_bench_params_hal()
        .into_iter()
        .filter(|p| p.n as u64 == bin_fhe_n())
        .collect();
    let vmp_params_fft: Vec<_> = default_bench_params_vmp()
        .into_iter()
        .filter(|p| p.n as u64 == bin_fhe_n())
        .collect();
    bench_ops(&hal_ops_fft, hal_params_fft.as_slice(), "FFT64Ref/hal", c);
    bench_ops(&vmp_ops_fft, vmp_params_fft.as_slice(), "FFT64Ref/hal", c);
}

// ── Layer 2: Core ───────────────────────────────────────────────────────────

fn core(c: &mut Criterion) {
    use poulpy_bench::core::suites::standard_ops;

    let core_ops_ntt = standard_ops::<NTT, WallTime>();
    let core_params_ntt: Vec<_> = default_bench_params_core()
        .into_iter()
        .filter(|p| is_standard_n(p.n as u64))
        .collect();
    bench_ops(&core_ops_ntt, core_params_ntt.as_slice(), "NTT4x30Ref/core", c);

    let core_ops_fft = standard_ops::<FFT, WallTime>();
    let core_params_fft: Vec<_> = default_bench_params_core()
        .into_iter()
        .filter(|p| p.n as u64 == bin_fhe_n())
        .collect();
    bench_ops(&core_ops_fft, core_params_fft.as_slice(), "FFT64Ref/core", c);
}

// ── Layer 3: Scheme ─────────────────────────────────────────────────────────

fn ckks(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::ckks_standard_ops;

    let ckks_ops = ckks_standard_ops::<NTT, WallTime>();
    let ckks_params: Vec<_> = default_bench_params_ckks()
        .into_iter()
        .filter(|p| is_standard_n(p.n as u64))
        .collect();
    bench_ops(&ckks_ops, ckks_params.as_slice(), "NTT4x30Ref/ckks", c);
}

fn bin_fhe(c: &mut Criterion) {
    use poulpy_bench::schemes::suites::bin_fhe_standard_ops;

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
    targets = hal, core, ckks, bin_fhe
}

criterion_main!(benches);
