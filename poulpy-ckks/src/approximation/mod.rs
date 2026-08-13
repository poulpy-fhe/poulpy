//! Reusable host-side polynomial approximation planning.
//!
//! This module turns smooth scalar functions into Chebyshev minimax
//! polynomials over one interval or a disjoint union, selects degrees against
//! precision or homomorphic-depth goals, and builds composite sign
//! approximations. Prepared ciphertext evaluation is exposed through
//! [`CKKSApproximationOps`] and [`PolynomialApproximation`].

mod remez;
mod select;
mod sign;

pub use crate::{api::CKKSApproximationOps, layouts::PolynomialApproximation};
pub use remez::{Minimax, Parity, RemezOptions, minimax, minimax_multi_interval, minimax_multi_interval_with, minimax_with};
pub use select::{
    DegreeChoice, degree_for_precision, degree_for_precision_multi_interval, degree_for_precision_multi_interval_with,
    degree_for_precision_with, error_bits, precision_at_depth, precision_at_depth_multi_interval,
    precision_at_depth_multi_interval_with, precision_at_depth_with,
};
pub use sign::{sign_composite_coeffs, sign_composite_coeffs_with_margin};
