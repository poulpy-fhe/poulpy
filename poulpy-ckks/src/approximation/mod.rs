//! Reusable host-side polynomial approximation planning.
//!
//! This module turns smooth scalar functions into Chebyshev minimax
//! polynomials, selects degrees against precision or homomorphic-depth goals,
//! and builds composite sign approximations. Prepared ciphertext evaluation is
//! exposed through [`CKKSApproximationOps`] and [`PolynomialApproximation`].

mod remez;
mod select;
mod sign;

pub use crate::{api::CKKSApproximationOps, layouts::PolynomialApproximation};
pub use remez::{Minimax, Parity, RemezOptions, minimax, minimax_with};
pub use select::{
    DegreeChoice, degree_for_precision, degree_for_precision_with, error_bits, precision_at_depth, precision_at_depth_with,
};
pub use sign::{sign_composite_coeffs, sign_composite_coeffs_with_margin};
