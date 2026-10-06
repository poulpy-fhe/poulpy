//! Blanket implementations connecting `poulpy-core` traits to the matching
//! backend family impl traits on [`poulpy_hal::layouts::Module`].
//!
//! Delegates enforce component noise metadata policy while wiring the safe public
//! traits to the backend-owned high-level extension point.

mod automorphism;
mod ci_conversion;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod linear_transformation;
mod operations;
mod polynomial_evaluation;
mod sampling;
