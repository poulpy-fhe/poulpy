//! Shared infrastructure for writing Criterion benchmarks against any
//! `poulpy` backend (`poulpy-cpu-ref`, `-avx`, `-avx512`, `-arm`, ...).
//!
//! Backend crates don't define their own benchmark logic — they write a thin
//! `benches/*.rs` binary that picks a backend type and calls into the runners
//! and suites defined here.
//!
//! # Layered module organization
//!
//! Benchmarks are organized by layer, mirroring the crate stack they exercise:
//!
//! - [`hal`]: raw `poulpy-hal` trait operations (`vec_znx`, `vec_znx_dft`,
//!   `vec_znx_big`, `svp`, `vmp`, `convolution`, `reim`).
//! - [`core`]: `poulpy-core` GLWE/GGSW operations (encryption, decryption,
//!   keyswitch, automorphism, external product, tensoring).
//! - [`schemes`]: full scheme-level operations (`ckks`, `bin_fhe`).
//!
//! # The runner / `BenchOp` / suite pattern
//!
//! Every benchmarked operation is a **runner**: a
//! `fn(&mut Bencher<'_, M>, &P)` that sets up its inputs once and times the
//! operation via `bencher.iter(...)`, generic over the backend and scoped to
//! exactly the traits it needs. Each module groups its runners into a data
//! table of [`BenchOp`]s (e.g. `hal::suites::vec_znx_ops`) — plain data, not
//! yet wired to Criterion, so callers can filter, reorder, or merge tables
//! from different groups before running them. [`bench_ops`] drives a
//! `&[BenchOp<M, P>]` against a sweep of parameters `&[P]`, one Criterion
//! group per op. Some modules (`core`, `schemes`) additionally expose a
//! `bench_suite_*` convenience function that bundles the standard table and
//! calls [`bench_ops`] directly.
//!
//! # Params
//!
//! [`params`] holds the sweep-parameter structs each layer's runners take
//! (`HalSweepParms`, `CoreParams`, `CkksBenchParams`, ...) plus
//! `default_bench_params_*` functions giving every backend the same
//! reasonable default sweep, so results can be comparable across backends.
//!
//! # Adding a new backend
//!
//! Write a `benches/*.rs` binary in the backend crate that imports the
//! relevant `*_ops`/`bench_suite_*` functions from this crate and drives them
//! against the backend's own type — no changes needed here.

pub mod core;
pub mod hal;
pub mod params;
pub mod schemes;

use std::fmt::Display;

use criterion::{Bencher, BenchmarkId, Criterion, measurement::Measurement};

// #[cfg(any(feature = "core-bench", feature = "bin-fhe-bench", feature = "ckks-bench"))]
// type BenchHostBackend = poulpy_cpu_ref::FFT64Ref;

/// Return the shared Criterion configuration used by all bench binaries.
///
/// Uses 100 samples with a 5-second measurement budget per benchmark.
/// Fast benchmarks complete in ~5 s; for slow benchmarks whose single
/// iteration exceeds the per-sample budget Criterion automatically extends
/// the run to collect at least a few samples (it will never cut a sample
/// short), so scheme-level benchmarks (blind rotate, CBS) may take longer.
pub fn criterion_config() -> criterion::Criterion {
    criterion::Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
}

pub fn ckks_criterion_config() -> criterion::Criterion {
    criterion_config()
}

/// One named operation in a benchmark suite: a criterion group (`name`) and
/// the runner that gets swept over every `P` entry.
pub struct BenchOp<M: Measurement, P> {
    pub name: &'static str,
    pub runner: fn(&mut Bencher<'_, M>, &P),
}

/// Runs one criterion group per op in `ops`, each expanded over every entry
/// in `sweeps`. `label` distinguishes the run (e.g. a backend tag) inside
/// each op's group, so results from multiple variants don't collide.
pub fn bench_ops<M: Measurement, P: Display>(ops: &[BenchOp<M, P>], sweeps: &[P], label: &str, c: &mut Criterion<M>) {
    for op in ops {
        let mut group = c.benchmark_group(format!("{}/{}", label, op.name));
        for sweep in sweeps {
            group.bench_with_input(BenchmarkId::from_parameter(sweep), sweep, op.runner);
        }
        group.finish();
    }
}
