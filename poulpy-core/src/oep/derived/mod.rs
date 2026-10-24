//! Crate-private default implementations formed by composing core operations.
//!
//! Calls use the selected backend contracts, including their scratch queries.
//! Backends can override these defaults in the corresponding `*Impl` trait.

pub(crate) mod automorphism;
pub(crate) mod ci_conversion;
pub(crate) mod conversion;
pub(crate) mod decompression;
pub(crate) mod encryption;
pub(crate) mod external_product;
pub(crate) mod keyswitching;
pub(crate) mod operations;
pub(crate) mod polynomial_evaluation;
pub(crate) mod preparation;
pub(crate) mod structure;
