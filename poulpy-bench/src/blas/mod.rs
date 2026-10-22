//! Benchmarks of `poulpy-blas`: ciphertext by ciphertext and ciphertext by public matrix products, the transpose, the
//! row and column sums and the row rotation of a replicated grid, on one `d x d` block per operand.
//!
//! - [`params`]: [`BlasBenchParams`](params::BlasBenchParams) and the default sweep over the ring degree and the block
//!   dimension.
//! - [`matmul`]: the runners, one per operation and variant. Each builds its point once, with only the rotation keys
//!   (and the tensor key) its operation looks up, checks the radix and the first result against the cleartext, then
//!   times the call.
//! - [`suites`]: the `BenchOp` tables and the `bench_blas` functions of the full, standard and light tiers.
//!
//! The layer targets NTT backends: at the default radix of 52 the lazy sums of a product of two ciphertexts exceed
//! what the FFT64 model allows, which the runners' radix check reports.

pub mod matmul;
pub mod params;
pub mod suites;
