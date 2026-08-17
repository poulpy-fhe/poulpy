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
//! Each layer has its own `params` module (`hal::params`, `core::params`,
//! `schemes::params`) holding the sweep-parameter structs its runners take
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

/// One named operation in a benchmark suite: a criterion group (`layer`,
/// `name`) and the runner that gets swept over every `P` entry. `layer`
/// (`"hal"`, `"core"`, `"ckks"`, `"bin_fhe"`, ...) is a property of which
/// suite table the op came from, not of how it's run.
pub struct BenchOp<M: Measurement, P> {
    pub layer: &'static str,
    pub name: &'static str,
    pub runner: fn(&mut Bencher<'_, M>, &P),
}

/// The short name Criterion group labels use for a backend marker type
/// (e.g. `poulpy_cpu_ref::ntt4x30::NTT4x30Ref` -> `"NTT4x30Ref"`).
fn backend_name<BE: ?Sized>() -> &'static str {
    std::any::type_name::<BE>().rsplit("::").next().unwrap()
}

/// Runs one criterion group per op in `ops`, each expanded over every entry
/// in `sweeps`. `backend` — a zero-sized instance of the backend marker type
/// the ops were built against (e.g. `Ntt`/`Fft` in the `poulpy-cpu-ref`
/// benches) — names the run; it's a value purely so `BE` can be inferred
/// from it, letting callers skip turbofish entirely. Each op's own `layer`
/// field distinguishes it inside the run, so results from multiple variants
/// don't collide.
///
/// `ops` and `sweeps` are taken as iterators rather than slices, so callers
/// can feed them a suite's own `Vec`/array directly, or a lazy `.filter()`
/// chain, without first collecting into an intermediate collection.
pub fn bench_ops<'a, BE, M, P, O, S>(_backend: BE, ops: O, sweeps: S, c: &mut Criterion<M>)
where
    M: Measurement,
    P: Display,
    O: IntoIterator<Item = &'a BenchOp<M, P>>,
    S: IntoIterator<Item = P>,
    M: 'a,
    P: 'a,
{
    let backend = backend_name::<BE>();
    // `sweeps` is expanded once per op, so it must be replayable even though
    // an arbitrary iterator/generator is only good for a single pass.
    let sweeps: Vec<P> = sweeps.into_iter().collect();
    for op in ops {
        let mut group = c.benchmark_group(format!("{}/{}/{}", backend, op.layer, op.name));
        for sweep in &sweeps {
            group.bench_with_input(BenchmarkId::from_parameter(sweep), sweep, op.runner);
        }
        group.finish();
    }
}
