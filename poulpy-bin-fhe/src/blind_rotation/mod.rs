//! Blind rotation and programmable bootstrapping.
//!
//! This module provides the foundational GLWE blind-rotation primitive used
//! throughout the binary FHE scheme.  Blind rotation evaluates a function
//! `f : Z_{2N} -> T_q` encoded in a [`LookupTable`] on an encrypted index
//! provided by an LWE ciphertext, producing a fresh GLWE ciphertext whose
//! constant term decrypts to `f(dec(lwe))` (modulo rounding noise).
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`LookupTable`] | Encoded evaluation function, allocated from a [`LookUpTableLayout`] |
//! | [`BlindRotationKey`] | Raw (standard) bootstrapping key — one GGSW per LWE dimension |
//! | [`BlindRotationKeyCompressed`] | Seed-compressed form of the bootstrapping key |
//! | [`BlindRotationKeyPrepared`] | DFT-pre-processed form for fast on-line evaluation |
//! | [`BlindRotationKeyLayout`] | Plain-old-data descriptor for key dimensions |
//!
//! ## Algorithm Variants
//!
//! The trait [`BlindRotationExecute`] is implemented per algorithm marker.
//! Currently the only marker is [`CGGI`], which implements the
//! Chillotti-Gama-Georgieva-Izabachène (CGGI / TFHE) blind rotation using
//! GGSW external products.  Three execution paths are selected at runtime based
//! on the key distribution:
//!
//! - **Standard** (`BinaryFixed` / `BinaryProb`): Classic CGGI, one external
//!   product per LWE coefficient.
//! - **Block-binary** (`BinaryBlock`): Batched CGGI processing multiple
//!   coefficients per product, reducing the total number of DFT operations.
//! - **Block-binary extended**: Block-CGGI with an extended LUT domain
//!   (`extension_factor > 1`), splitting the lookup table across multiple
//!   polynomials to increase plaintext precision.
//!
//! ## Key Lifecycle
//!
//! Blind rotation keys follow the standard three-stage lifecycle:
//! 1. Allocate with `BlindRotationKey::alloc`.
//! 2. Fill with `encrypt_sk` supplying the GLWE and LWE secret keys.
//! 3. Prepare with `BlindRotationKeyPrepared::prepare` before any evaluation.
//!
//! The compressed variant stores only a 32-byte seed for the mask component,
//! reducing serialised size at the cost of decompression during preparation.
mod algorithms;
mod encryption;
mod layouts;
mod lut;
mod utils;

pub use algorithms::*;
pub use encryption::*;
pub use layouts::*;
pub use lut::*;
#[cfg(all(test, feature = "enable-bin-fhe"))]
pub mod tests;
