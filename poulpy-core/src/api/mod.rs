//! Safe, user-facing trait definitions for Module-LWE operations.
//!
//! Traits are organized by operation family:
//! - `automorphism` -- Galois automorphisms on ciphertexts and automorphism keys.
//! - `bytes_of` -- backend-routed byte sizes for core layouts.
//! - `conversion` -- conversions between ciphertext representations.
//! - `decryption` -- secret-key decryption operations.
//! - `encryption` -- secret/public-key encryption and evaluation-key generation.
//! - `external_product` -- GLWE/GGLWE/GGSW external products.
//! - `keyswitching` -- LWE/GLWE/GGLWE/GGSW key-switching.
//! - `noise` -- runtime noise measurement helpers for ciphertexts.
//! - `operations` -- arithmetic helpers, packing, trace, and tensoring.
//!
//! Scheme authors can program against these traits directly. Execution is
//! dispatched through the [`crate::oep`] backend extension points by blanket
//! implementations in the (private) `delegates` module.

mod automorphism;
mod bytes_of;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod linear_transformations;
mod noise;
mod operations;
mod polynomial_evaluation;
mod transfer;

pub use automorphism::*;
pub use bytes_of::*;
pub use conversion::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use linear_transformations::*;
pub use noise::*;
pub use operations::*;
pub use polynomial_evaluation::*;
pub use transfer::*;
