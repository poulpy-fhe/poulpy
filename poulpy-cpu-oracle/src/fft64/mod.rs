//! Scalar correctness oracle using scalar f64 FFT.

mod module;
mod reim;
mod znx;

pub use reim::FFT64ReimTable;

/// Scalar correctness oracle using scalar f64 FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64Oracle;
