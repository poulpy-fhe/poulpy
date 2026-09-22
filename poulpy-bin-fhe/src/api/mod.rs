//! Backend-agnostic, user-facing binary-FHE operation contracts.
//!
//! - [`blind_rotation`]: encrypted LUT evaluation, key lifecycle, and LUT helpers.
//! - [`circuit_bootstrapping`]: constant/exponent output, reusable plans, and keys.
//! - [`bdd`]: encrypted selection and integer-circuit evaluation.
//!
//! Import traits from their family or directly from this module. All public
//! operation traits delegate through [`crate::oep`], including LUT construction
//! and compressed-key allocation. Algorithm and layout modules re-export the
//! established names for compatibility.
pub mod bdd;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
pub use bdd::*;
pub use blind_rotation::*;
pub use circuit_bootstrapping::*;
