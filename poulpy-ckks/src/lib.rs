#![allow(private_bounds)]

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
//! unnormalized traits ([`api::CKKSAddOpsUnnormalized`] and
//! [`api::CKKSSubOpsUnnormalized`]) write into an
//! [`layouts::UnnormalizedCKKSCiphertext`] for callers who want to fuse
//! several linear steps before normalizing explicitly. Limb digits in that
//! wrapper may hold un-propagated carries (wider than `base2k` bits), so
//! passing it to any DFT-domain primitive (keyswitching, convolution,
//! automorphisms) would produce incorrect decryptions. The wrapper does not
//! implement [`GLWEToBackendRef`] or [`GLWEToBackendMut`], making such misuse
//! a compile error. Call [`layouts::UnnormalizedCKKSCiphertext::normalize`]
//! before the next keyswitching or convolution step.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`encoding`] | CKKS encoders/decoders, including slot-wise real/imaginary packing |
//! | [`layouts`] | CKKS ciphertext/plaintext wrappers and metadata-aware allocation helpers |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | bootstrapping | Planned CKKS bootstrapping |

use poulpy_core::layouts::{Base2K, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::Backend;

pub mod api;
pub mod default;
pub(crate) mod delegates;
pub mod encoding;
mod error;
pub mod layouts;
pub mod leveled;
pub mod oep;
pub mod test_suite;
pub use error::CKKSCompositionError;
pub(crate) use error::{
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget, ensure_base2k_match,
    ensure_plaintext_alignment, ensure_plaintext_coeff_in_range, ensure_plaintext_degree_match,
};

pub type CKKSCiphertextRef<'a, BE> = layouts::CKKSCiphertext<<BE as Backend>::BufRef<'a>>;
pub type CKKSCiphertextMut<'a, BE> = layouts::CKKSCiphertext<<BE as Backend>::BufMut<'a>>;

pub trait CKKSPlaintextToBackendRef<BE: Backend>: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos {}

impl<BE: Backend, T> CKKSPlaintextToBackendRef<BE> for T where T: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos {}

/// Marker bound for CKKS ciphertext type parameters.
///
/// Combines [`GLWEInfos`] (which already implies [`poulpy_core::layouts::LWEInfos`])
/// with [`CKKSInfos`] to collapse the repeated `GLWEInfos + CKKSInfos` pair found
/// throughout the API into a single, named constraint.
pub trait CKKSCtBounds: GLWEInfos + CKKSInfos {}

impl<T: GLWEInfos + CKKSInfos> CKKSCtBounds for T {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// CKKS semantic precision metadata carried by ciphertexts and plaintexts.
///
/// `log_delta` is the scaling precision of the encoded value and
/// `log_budget` is the remaining homomorphic headroom available above `log_delta`.
pub struct CKKSMeta {
    /// Base 2 logarithm of the decimal precision.
    pub log_delta: usize,
    /// Base 2 logarithm of the remaining homomorphic capacity.
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

    /// Returns the next multiple of [`Base2K`] greater than [`Self::log_delta`] + [`Self::log_budget`].
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

/// Mutable CKKS metadata access for ciphertext/plaintext containers.
pub trait SetCKKSInfos: CKKSInfos {
    /// Replaces the semantic CKKS metadata.
    fn set_meta(&mut self, meta: CKKSMeta);

    /// Updates only the base-2 logarithm of the encoded scaling factor.
    fn set_log_delta(&mut self, log_delta: usize) {
        let mut meta = self.meta();
        meta.log_delta = log_delta;
        self.set_meta(meta);
    }

    /// Updates only the base-2 logarithm of the remaining homomorphic budget.
    fn set_log_budget(&mut self, log_budget: usize) {
        let mut meta = self.meta();
        meta.log_budget = log_budget;
        self.set_meta(meta);
    }
}

pub(crate) fn ckks_offset_binary<R, A, B>(res: &R, a: &A, b: &B) -> usize
where
    R: LWEInfos + CKKSInfos + ?Sized,
    A: LWEInfos + CKKSInfos + ?Sized,
    B: LWEInfos + CKKSInfos + ?Sized,
{
    a.effective_k().min(b.effective_k()).saturating_sub(res.max_k().as_usize())
}

pub(crate) fn ckks_offset_unary<R, A>(res: &R, a: &A) -> usize
where
    R: LWEInfos + CKKSInfos + ?Sized,
    A: LWEInfos + CKKSInfos + ?Sized,
{
    a.effective_k().saturating_sub(res.max_k().as_usize())
}
