//! Benchmarks of `poulpy-libm`: the evaluation of prepared function plans.
//!
//! - [`suites`]: `bench_libm`, `exp`, `log` and `pow` on one backend, each plan
//!   built and uploaded once before timing.
//!
//! The layer targets NTT backends: its parameter set has a radix of 52.

pub mod suites;
