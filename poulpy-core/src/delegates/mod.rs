//! Blanket implementations connecting `poulpy-core` traits to the matching
//! backend family impl traits on [`poulpy_hal::layouts::Module`].
//!
//! Delegates maintain encryption provenance while wiring the safe public
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

/// Arithmetic keeps shared provenance only when both inputs agree.
fn matching_metadata(
    a: Option<crate::EncryptionMetadata>,
    b: Option<crate::EncryptionMetadata>,
) -> Option<crate::EncryptionMetadata> {
    a.filter(|metadata| b == Some(*metadata))
}
