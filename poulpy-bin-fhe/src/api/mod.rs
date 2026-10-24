//! Backend-agnostic, user-facing binary-FHE operation contracts.
//!
//! Operation traits cover encrypted LUT evaluation and key lifecycle (blind
//! rotation), constant/exponent output and reusable plans (circuit
//! bootstrapping), and encrypted selection and integer-circuit evaluation
//! (BDD). Import them from this module. All public operations, LUT
//! construction included, delegate through [`crate::oep`]; key allocation and
//! sizing are direct on the module. Algorithm and layout modules re-export the
//! established names for compatibility.
mod bdd;
mod blind_rotation;
mod circuit_bootstrapping;
pub use bdd::*;
pub use blind_rotation::*;
pub use circuit_bootstrapping::*;
