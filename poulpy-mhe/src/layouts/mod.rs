//! Public aggregatable transcript (PAT) layouts, one per transcript shape.
//!
//! A seeded PAT (`*Compressed`) stores bodies whose masks every party
//! regenerates from the common seed; an unseeded PAT stores the full
//! ciphertext. Unseeded GLWE transcripts are plain core
//! [`GLWE`](poulpy_core::layouts::GLWE)s, which already carry the canonical flag.

mod alloc;
mod gglwe_pat;
mod gglwe_pat_compressed;
mod glwe_automorphism_key_pat_compressed;
mod glwe_pat_compressed;
mod glwe_switching_key_pat_compressed;

pub use alloc::*;
pub use gglwe_pat::*;
pub use gglwe_pat_compressed::*;
pub use glwe_automorphism_key_pat_compressed::*;
pub use glwe_pat_compressed::*;
pub use glwe_switching_key_pat_compressed::*;
