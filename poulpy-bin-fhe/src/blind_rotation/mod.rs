//! Blind rotation and programmable bootstrapping.
//!
//! This module provides the foundational GLWE blind-rotation primitive used
//! throughout the binary FHE scheme.  Blind rotation evaluates a function
//! `f : Z_{2N} -> T_q` encoded in a [`LookupTable<AlignedBuf, i64>`] on an encrypted index
//! provided by an LWE ciphertext, producing a fresh GLWE ciphertext whose
//! constant term decrypts to `f(dec(lwe))` (modulo rounding noise).
//!
//! ## User-facing API
//!
//! Import operation traits from [`crate::api::blind_rotation`] for module calls
//! and their scratch queries. The API documentation includes a generic usage
//! example. Key and plan convenience methods dispatch through the same traits.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`LookupTable<AlignedBuf, i64>`] | Encoded evaluation function, allocated from a [`LookUpTableLayout`] |
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
//! The compressed variant stores the body and per-element mask seeds. Decompress
//! it through [`crate::api::BlindRotationKeyDecompress`] before preparation.
//!
//! Backend implementations explicitly select operations through [`crate::oep`].
//! The callable compositions live in [`crate::reference::blind_rotation`]. CGGI
//! is an algorithm marker, independent of the backend and execution schedule.
//! Parallel block scheduling is an explicit backend selection; the canonical
//! composition always accumulates contributions in coefficient order.
mod algorithms;
mod encryption;
pub(crate) mod host_znx;
mod layouts;
mod lut;
pub(crate) mod utils;

pub use algorithms::*;
pub use encryption::*;
pub use layouts::*;
pub use lut::*;
/// Backend-generic test bodies, instantiated by the backend crates through
/// [`bin_fhe_backend_test_suite!`](crate::bin_fhe_backend_test_suite).
pub mod test_suite;

#[cfg(test)]
mod serialization_tests;
