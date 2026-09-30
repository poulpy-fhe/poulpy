//! Scalar correctness oracle using scalar NTT over four 30-bit primes.

mod module;
mod prim;
mod vec_znx_big;
mod znx;

pub use module::NTT4x30OracleHandle;

/// Scalar correctness oracle using scalar NTT over four 30-bit primes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30Oracle;
