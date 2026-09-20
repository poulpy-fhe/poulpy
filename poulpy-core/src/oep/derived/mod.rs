//! Default implementations formed by composing core operations.
//!
//! Calls use the selected backend contracts, including their scratch queries.
//! Backends can override these defaults in the corresponding `*Impl` trait.

pub mod automorphism;
pub mod conversion;
pub mod encryption;
pub mod external_product;
pub mod keyswitching;
pub mod operations;
pub mod polynomial_evaluation;
pub mod structure;
