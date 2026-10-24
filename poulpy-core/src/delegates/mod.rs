//! Blanket implementations connecting `poulpy-core` traits to the matching
//! backend family impl traits on [`poulpy_hal::layouts::Module`].
//!
//! Delegates forward the safe public traits to backend extension points.
//! Reference and derived implementations own component noise metadata updates.

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
