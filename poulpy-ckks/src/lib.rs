//! # poulpy-ckks
//!
//! Backend-agnostic implementation of the CKKS (Cheon-Kim-Kim-Song)
//! homomorphic encryption scheme, built on top of the low-level primitives
//! provided by `poulpy-core`, `poulpy-hal`, and the available compute
//! backends (`poulpy-cpu-ref`, `poulpy-cpu-avx`).
//!
//! The crate uses a bivariate polynomial representation over the Torus
//! (base-`2^{base2k}` digits) instead of the RNS representation used by
//! most other CKKS libraries. Public precision management is exposed through
//! [`CKKSMeta`]:
//!
//! - `log_delta`: base-2 logarithm of the encoded plaintext scaling factor
//! - `log_budget`: remaining homomorphic headroom, also tracked in bits
//!
//! Together they define the semantic torus width of a value:
//! `effective_k() = log_delta + log_budget`.
//! Storage is rounded up to the next multiple of `base2k`, so the allocated
//! width `max_k()` may exceed `effective_k()`. Arithmetic APIs update this
//! metadata for you, while maintenance helpers let you compact or resize owned
//! buffers without violating those invariants.
//!
//! Safe add/sub operations return K-normalized ciphertexts. The paired
//! `unsafe` traits ([`leveled::CKKSAddOpsUnsafe`] and
//! [`leveled::CKKSSubOpsUnsafe`]) expose `*_unsafe` variants for callers who
//! want to fuse several linear steps before normalizing explicitly. The
//! current `examples/poly2.rs` demonstrates that style: it uses unsafe
//! intermediate linear ops, calls `glwe_normalize_assign` before the ct-ct
//! multiply, and finishes with a normalized fused `mul_add`.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`encoding`] | CKKS encoders/decoders, including slot-wise real/imaginary packing |
//! | [`layouts`] | CKKS ciphertext/plaintext wrappers and metadata-aware allocation helpers |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | bootstrapping | Planned CKKS bootstrapping |

use poulpy_core::layouts::{Base2K, TorusPrecision};

pub mod encoding;
mod error;
pub mod layouts;
pub mod leveled;
pub mod oep;
pub use error::CKKSCompositionError;
pub(crate) use error::{checked_log_budget_sub, checked_mul_ct_log_budget, ensure_base2k_match, ensure_plaintext_alignment};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// CKKS semantic precision metadata carried by ciphertexts and plaintexts.
///
/// `log_delta` is the scaling precision of the encoded value and
/// `log_budget` is the remaining homomorphic headroom available before the
/// value must be rescaled or truncated.
pub struct CKKSMeta {
    /// Base 2 logarithm of the decimal precision.
    pub log_delta: usize,
    /// Base 2 logarithm of the Remaining homomorphic capacity.
    pub log_budget: usize,
}

/// Common metadata accessors for CKKS ciphertext and plaintext containers.
///
/// This trait exposes the semantic precision of a value independently from the
/// raw limb storage used by the underlying torus representation.
pub trait CKKSInfos {
    /// Returns the complete metadata pair.
    fn meta(&self) -> CKKSMeta;

    /// Returns the base-2 logarithm of the encoded decimal scaling factor.
    fn log_delta(&self) -> usize;

    /// Returns the base-2 logarithm of the remaining homomorphic capacity.
    fn log_budget(&self) -> usize;

    /// Returns the next multiple of [Base2K] greater than [Self::log_delta] + [Self::log_budget].
    fn min_k(&self, base2k: Base2K) -> TorusPrecision {
        ((self.log_delta() + self.log_budget()).next_multiple_of(base2k.as_usize())).into()
    }

    /// Returns the semantic torus width carried by the value.
    ///
    /// This is `log_delta + log_budget` and may differ from the rounded
    /// storage capacity `max_k()`.
    fn effective_k(&self) -> usize {
        self.log_delta() + self.log_budget()
    }
}

impl CKKSInfos for CKKSMeta {
    fn meta(&self) -> CKKSMeta {
        *self
    }

    fn log_delta(&self) -> usize {
        self.log_delta
    }

    fn log_budget(&self) -> usize {
        self.log_budget
    }
}
