//! Standalone CKKS bootstrapping run, for profiling.
//!
//! Compiles the end-to-end bootstrapping pipeline (the `ntt120_f64` reference
//! backend — same composition as `tests::ckks_tests::ntt120_f64::bootstrapping_e2e`)
//! into a single binary so it can be run under a sampling profiler and explored
//! in the browser: every function, its call count and timings, flame graph and
//! timeline.
//!
//! Build with optimizations **and** debug symbols (so the profiler resolves
//! function names) and record with [`samply`](https://github.com/mstange/samply):
//!
//! ```text
//! cargo install samply        # once
//! cargo build -p poulpy-cpu-ref --example bootstrap_trace --features enable-ckks --profile profiling
//! samply record ./target/profiling/examples/bootstrap_trace
//! ```
//!
//! `samply` captures the run and opens the Firefox Profiler in your browser
//! (call tree, per-function self/total time, flame graph, timeline). Set
//! `BOOTSTRAP_ITERS=N` to run the pipeline N times for more samples.
//!
//! Alternatives that also land in a browser:
//! - `perf record -g --call-graph dwarf -- ./target/profiling/examples/bootstrap_trace`
//!   then load `perf.data` at <https://profiler.firefox.com>.
//! - `cargo flamegraph -p poulpy-cpu-ref --example bootstrap_trace --features enable-ckks`
//!   (SVG, open in any browser).

use poulpy_ckks::test_suite::{NTT120_PARAMS_F64, bootstrapping::test_bootstrapping_standard_e2e};
use poulpy_cpu_ref::{
    FFT64ReimTable, NTT120Ref,
    layouts::{HostBytesBackend, Module},
};

fn main() {
    let params = NTT120_PARAMS_F64;

    // `test_bootstrapping_e2e` builds its own modules internally; these only
    // satisfy the signature (they are ignored by the test body).
    let module = Module::<NTT120Ref>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);

    test_bootstrapping_standard_e2e::<NTT120Ref, f64, FFT64ReimTable<f64>>(params, &module, &host_module);
}
