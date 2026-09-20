//! Portable core algorithms built from HAL operations.
//!
//! The `*Reference` methods and free functions in this module are reusable
//! implementations. Backend `*Impl` traits explicitly select them for public
//! dispatch. Compositions of other core operations live in
//! [`crate::oep::derived`] and provide defaults on those backend traits.
//!
//! Overrides must implement the same circuit. Parity tests use a caller-selected
//! validated backend and compare integer results and metadata on identical
//! inputs; randomized operations additionally control the sampled values.

pub mod automorphism;
pub mod conversion;
pub mod decryption;
pub mod encryption;
pub mod external_product;
pub mod keyswitching;
pub mod linear_transformation;
pub mod noise;
pub mod operations;
pub mod polynomial_evaluation;
