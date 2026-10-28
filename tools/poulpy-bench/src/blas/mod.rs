//! Benchmarks of `poulpy-blas`: ciphertext by ciphertext and ciphertext by public matrix products, the transpose, the
//! row and column sums and the row rotation of a replicated grid, on one `d x d` block per operand.
//!
//! - [`params`]: [`BlasBenchParams`](params::BlasBenchParams) and the default sweep over the ring degree and the block
//!   dimension.
//! - [`matmul`]: the runners of the products and the transpose, one per operation and variant. Each builds its point
//!   once, with only the rotation keys (and the tensor key) its operation looks up, checks the radix and the first
//!   result against the cleartext, then times the call.
//! - [`intra`]: the runners of the row and column sums and of the row rotation of a replicated grid, built alike.
//! - [`suites`]: the `BenchOp` tables and the `bench_blas` functions of the full, standard and light tiers.
//! - [`scale`]: large blocks, outside Criterion: the radix table of every product, and checks on grids of many
//!   blocks, returning those with an error beyond the bound or a radix the model rejects.
//!
//! The layer targets NTT backends: at the default radix of 52 the lazy sums of a product of two ciphertexts exceed
//! what the FFT64 model allows, which the runners' radix check reports.

pub mod intra;
pub mod matmul;
pub mod params;
pub mod scale;
pub mod suites;
