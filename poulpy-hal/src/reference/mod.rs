//! Portable reference kernels shared by every backend.
//!
//! Scalar polynomial arithmetic over plain `[i64]` slices, with no backend
//! layout or dispatch involved. Backends build their accelerated kernels on
//! these and the test suites use them as the correctness oracle.

pub mod znx;
