//! Decryption of ciphertexts using secret keys.
//!
//! This module provides traits and implementations for decrypting
//! lattice-based ciphertexts back into plaintexts:
//!
//! - [`GLWEDecrypt`]: decrypts a GLWE ciphertext using a prepared GLWE secret key.
//! - [`GLWETensorDecrypt`]: decrypts a GLWE tensor ciphertext using both a
//!   standard GLWE secret key and a tensor secret key.
//! - [`LWEDecrypt`]: decrypts an LWE ciphertext using an LWE secret key.
//!
//! Each trait exposes a scratch-bytes query method and a decryption method.
//! Scratch space must be pre-allocated by the caller using the corresponding
//! `*_tmp_bytes` function.

pub mod glwe;
pub mod glwe_tensor;
pub mod lwe;

pub(crate) use glwe::glwe_decrypt_backend_inner;
pub use glwe::*;
