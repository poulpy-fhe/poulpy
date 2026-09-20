//! Backend-agnostic Module-LWE-based homomorphic encryption primitives.
//!
//! `poulpy-core` implements the cryptographic building blocks of a
//! Module-LWE (MLWE) fully homomorphic encryption (FHE) scheme on top
//! of the hardware-abstraction layer provided by [`poulpy_hal`].
//! The public operation traits live in [`api`], while their blanket
//! implementations on [`poulpy_hal::layouts::Module<BE>`] delegate to
//! backend extension points in [`oep`]. This makes the crate portable
//! across CPU, AVX, and future backends. [`reference`](mod@crate::reference)
//! contains reusable algorithms built from HAL operations. [`oep::derived`]
//! contains defaults built from other core operations, preserving their backend
//! dispatch. Backends explicitly implement `*Impl` traits to select or replace
//! these algorithms. Tests compare integer results and metadata with a portable
//! CPU reference execution. Sampling has a distribution and reproducibility
//! contract; randomized parity uses controlled samples.
//!
//! # Architecture
//!
//! ## Three-layer layout system
//!
//! Every ciphertext and key type exists in three representations,
//! defined in the [`layouts`] module:
//!
//! | Layer | Purpose | Generic params |
//! |---|---|---|
//! | **Standard** (`GLWE<D>`, `GGSW<D>`, ...) | Serializable, platform-independent | `D: Data` |
//! | **Compressed** (`GLWECompressed<D>`, ...) | Reduced storage via seed-based mask regeneration | `D: Data` |
//! | **Prepared** (`GLWEPrepared<D, B>`, ...) | DFT-domain, optimised for fast polynomial arithmetic | `D: Data, B: Backend` |
//!
//! `D: Data` abstracts ownership: `AlignedBuf` (owned), `&[u8]` (borrowed),
//! or `&mut [u8]` (mutable borrow).
//!
//! ## Scratch-space allocation
//!
//! Arithmetic operations accept a [`poulpy_hal::layouts::ScratchArena`] for
//! their temporary backend storage, allocated via [`ScratchArenaTakeCore`].
//! The corresponding `*_tmp_bytes` query advertises the required arena capacity.
//! Higher-level planning and polynomial-evaluation policies can own collections
//! and allocate separately; the arena contract does not prohibit those allocations.
//!
//! ## Parameter newtypes
//!
//! Domain quantities are wrapped in [`u32`]-backed newtypes with
//! saturating arithmetic: [`layouts::Degree`], [`layouts::Base2K`],
//! [`layouts::TorusPrecision`], [`layouts::Rank`], [`layouts::Dnum`],
//! [`layouts::Dsize`].
//!
//! # Ciphertext types
//!
//! * [`layouts::LWE`] -- Learning With Errors ciphertext (scalar ring element).
//! * [`layouts::GLWE`] -- Generalised LWE ciphertext (polynomial ring).
//! * [`layouts::GGLWE`] -- Gadget GLWE, a matrix of GLWE rows used for key-switching.
//! * [`layouts::GGSW`] -- Gadget GSW ciphertext, used as the left operand of external products.
//!
//! # Module overview
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`api`] | Safe, user-facing operation traits |
//! | [`layouts`] | Type definitions for all ciphertext, key, plaintext, and secret layouts |
//! | encryption | Secret-key, public-key, and compressed encryption |
//! | decryption | Decryption of GLWE, LWE, and tensor ciphertexts |
//! | operations | Ciphertext arithmetic (add, sub, rotate, shift, normalize, mul) |
//! | external\_product | GGSW x GLWE external product (core HE multiplication) |
//! | keyswitching | Key-switching for GLWE, GGLWE, GGSW, and LWE |
//! | automorphism | Galois automorphisms on ciphertexts and keys |
//! | conversion | LWE / GLWE and GGLWE -> GGSW conversions |
//! | glwe\_packer | On-the-fly GLWE packing with O(log N) memory |
//! | glwe\_packing | HashMap-based GLWE slot packing |
//! | glwe\_trace | Normalized GLWE trace (average of automorphisms) |
//! | noise | Noise-variance estimation for parameter selection |
//! | dist | Secret-key distribution descriptors |
//! | scratch | Arena-style scratch allocation for ciphertext temporaries |

pub mod api;
mod delegates;
mod dist;
pub mod error;
pub mod oep;
pub mod reference;
mod scratch;
mod utils;

pub mod layouts;
pub use api::*;
pub use dist::*;
pub use error::{CoreError, Result};
pub use reference::encryption::*;
pub use reference::linear_transformation::*;
pub(crate) use reference::noise::{log2_std_noise_glwe_tensor, log2_std_noise_glwe_tensor_relinearized};
pub use reference::operations::*;
pub use reference::polynomial_evaluation::{BSGSOps, GiantStepTensorBounds};
pub use scratch::*;

pub(crate) mod decryption {
    pub(crate) use crate::reference::decryption::*;
}

pub(crate) mod encryption {
    pub(crate) mod gglwe {
        pub(crate) use crate::reference::encryption::gglwe::*;
    }

    pub(crate) mod glwe {}

    pub(crate) mod glwe_switching_key {
        pub(crate) use crate::reference::encryption::glwe_switching_key::*;
    }

    pub(crate) use crate::reference::encryption::*;
}

pub(crate) mod noise {
    pub(crate) mod glwe {
        pub(crate) use crate::reference::noise::glwe::*;
    }

    pub(crate) use crate::reference::noise::*;
}

pub(crate) mod operations {}

pub mod test_suite;

#[cfg(test)]
mod tests;
