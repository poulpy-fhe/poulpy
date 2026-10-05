//! Public aggregatable transcript (PAT) layouts, one per transcript shape, and
//! the protocol shares built on them.
//!
//! A seeded PAT (`*Compressed`) stores bodies whose masks every party
//! regenerates from the common seed; an unseeded PAT stores the full
//! ciphertext. Unseeded GLWE transcripts are plain core
//! [`GLWE`](poulpy_core::layouts::GLWE)s. A protocol share (`*Share`) wraps
//! the PATs of one party's contribution with the metadata of its result.

mod alloc;
mod gglwe_pat;
mod gglwe_pat_compressed;
mod ggsw_share;
mod glwe_automorphism_key_share;
mod glwe_keyswitch_share;
mod glwe_pat_compressed;
mod glwe_public_key_share;
mod glwe_share;
mod glwe_sharing_share;
mod glwe_switching_key_share;
mod glwe_tensor_key_share;

pub use alloc::*;
pub use gglwe_pat::*;
pub use gglwe_pat_compressed::*;
pub use ggsw_share::*;
pub use glwe_automorphism_key_share::*;
pub use glwe_keyswitch_share::*;
pub use glwe_pat_compressed::*;
pub use glwe_public_key_share::*;
pub use glwe_sharing_share::*;
pub use glwe_switching_key_share::*;
pub use glwe_tensor_key_share::*;
