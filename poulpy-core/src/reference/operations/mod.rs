//! Arithmetic operations on GLWE and GGSW ciphertexts.
//!
//! This module provides traits and implementations for element-wise
//! arithmetic on ciphertexts of either [ring](crate::layouts#rings),
//! including addition, subtraction, rotation (standard ring only),
//! right-shift, normalization, copy, multiplication by a constant
//! polynomial, multiplication by a plaintext GLWE, and tensor product.
//!
//! Operations are available in both out-of-place (`res = op(a, b)`) and
//! in-place (`res = op(res, a)`) variants. In-place variants may require
//! a scratch buffer.

mod gadget;
mod glwe;
mod glwe_mul_plain_sum;

pub use gadget::*;
pub use glwe::*;
pub use glwe_mul_plain_sum::*;
