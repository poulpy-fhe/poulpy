//! Arithmetic operations on GLWE and GGSW ciphertexts.
//!
//! This module provides traits and implementations for element-wise
//! arithmetic on ciphertexts in the polynomial ring `Z[X]/(X^N+1)`,
//! including addition, subtraction, rotation (automorphism),
//! right-shift, normalization, copy, multiplication by a constant
//! polynomial, multiplication by a plaintext GLWE, and tensor product.
//!
//! Operations are available in both out-of-place (`res = op(a, b)`) and
//! in-place (`res = op(res, a)`) variants. In-place variants may require
//! a scratch buffer.

mod ggsw;
mod glwe;

pub use ggsw::*;
pub use glwe::*;
