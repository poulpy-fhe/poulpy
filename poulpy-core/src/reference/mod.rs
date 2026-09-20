//! Portable implementations of core operation semantics.
//!
//! Bodies compose HAL operations and other core contracts. They are independently
//! callable free functions or `*Composition` methods, selected for a backend through
//! the explicit forwarding macros in [`crate::oep`]. Abstract `*Reference` traits
//! are override contracts; the separately documented tensoring reference trait is
//! a composition helper behind its directly implemented backend extension point.
//!
//! An optimized implementation is checked against an explicit portable
//! `poulpy-cpu-ref` execution on identical logical inputs. Tests compare canonical
//! results and metadata rather than backend-dependent prepared storage. Randomized
//! compositions additionally require controlled samples, because backend random
//! streams are not required to match.

pub mod automorphism;
pub mod conversion;
pub mod decryption;
pub mod encryption;
pub mod external_product;
pub mod glwe_packing;
pub mod glwe_trace;
pub mod keyswitching;
pub mod linear_transformation;
pub mod noise;
pub mod operations;
pub mod polynomial_evaluation;
