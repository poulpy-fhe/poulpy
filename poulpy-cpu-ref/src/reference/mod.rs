//! Pure-Rust reference implementations of all polynomial operations.
//!
//! Contains scalar polynomial arithmetic (`znx`), vector-level operations
//! (`vec_znx`), and an FFT64 implementation (`fft64`). Used as a
//! correctness oracle for backend testing via the
//! [`poulpy_hal::test_suite`] module.

pub mod fft64;
pub mod ntt120;
pub mod vec_znx;
pub mod znx;
