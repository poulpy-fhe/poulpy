//! Independently callable binary-FHE algorithms built on core and HAL.
//!
//! These bodies define the circuits reproduced by backend overrides. Same-layer
//! compositions are crate-private derived defaults on the OEP contracts.
pub mod bdd;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
