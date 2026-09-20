//! Portable core algorithms built from HAL operations.
//!
//! The `*Reference` methods and free functions in this module are reusable
//! implementations. Backend `*Impl` traits explicitly select them for public
//! dispatch. Compositions of other core operations live in
//! [`crate::oep::derived`] and provide defaults on those backend traits.
//!
//! Optimized implementations are checked against an explicit portable CPU
//! execution on identical logical inputs. Tests compare integer results and
//! metadata; randomized operations additionally control the sampled values.

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
