//! Standalone CKKS bootstrapping run, for profiling.
//!
//! Compiles the end-to-end bootstrapping pipeline (the `ntt4x30_f64` reference
//! backend — same composition as `tests::ckks_tests::ntt4x30_f64::bootstrapping_e2e`)
//! into a single binary so it can be run under a sampling profiler and explored
//! in the browser: every function, its call count and timings, flame graph and
//! timeline.
//!
//! Build with optimizations **and** debug symbols (so the profiler resolves
//! function names) and record with [`samply`](https://github.com/mstange/samply):
//!
//! ```text
//! cargo install samply        # once
//! cargo build -p poulpy-cpu-portable --example bootstrap_trace --features enable-ckks --profile profiling
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
//! - `cargo flamegraph -p poulpy-cpu-portable --example bootstrap_trace --features enable-ckks`
//!   (SVG, open in any browser).

use poulpy_ckks::test_suite::{BASE52_PARAMS_F64, bootstrapping::test_bootstrapping_standard_e2e};
use poulpy_cpu_portable::{FFT64ReimTable, NTT4x30Portable};

fn main() {
    test_bootstrapping_standard_e2e::<NTT4x30Portable, f64, FFT64ReimTable<f64>>(BASE52_PARAMS_F64);
}
