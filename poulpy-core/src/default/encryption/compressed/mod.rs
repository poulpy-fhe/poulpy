//! Compressed encryption for GLWE-based ciphertexts and evaluation keys.
//!
//! This module provides seed-compressed variants of encryption operations. Instead of
//! storing the full ciphertext mask (the `a` component), compressed ciphertexts store
//! only a PRNG seed (`seed_xa`) from which the mask is deterministically regenerated
//! during decompression. Only the body (the `b` component) is stored explicitly.
//!
//! This significantly reduces the storage and transmission cost of ciphertexts and
//! evaluation keys, since the mask typically dominates the overall size.
//!
//! Each compressed encryption trait mirrors its non-compressed counterpart but produces
//! a compressed output type and takes a `seed_xa` parameter for deterministic mask
//! generation, along with a `source_xe` PRNG source for sampling encryption noise.

mod gglwe;
mod gglwe_to_ggsw_key;
mod ggsw;
mod glwe_automorphism_key;
mod glwe_ct;
mod glwe_switching_key;
mod glwe_tensor_key;

pub use gglwe::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe_automorphism_key::*;
pub use glwe_ct::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
