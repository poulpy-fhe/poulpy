//! Benchmarks of `poulpy-libm`: the evaluation of prepared function plans on one ciphertext.
//!
//! - [`params`]: [`LibmBenchParams`](params::LibmBenchParams) and the default sweep over the ring degree.
//! - [`functions`]: the runners, one per operation. Each builds its point once: the plan on the host, transferred to
//!   the backend, a tensor key, a backend-filled input and destination, and the scratch the operation's query reports.
//! - [`suites`]: the `BenchOp` table and the `bench_libm` functions of the full, standard and light tiers.
//!
//! The layer targets NTT backends: its points have a radix of 52.

pub mod functions;
pub mod params;
pub mod suites;
