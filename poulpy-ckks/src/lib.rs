#![cfg_attr(not(all(feature = "libquadmath", target_arch = "x86_64")), feature(f128))]
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
//! `k() = log_delta + log_budget`.
//! Storage is rounded up to the next multiple of `base2k`, so the allocated
//! width `k()` may exceed `k()`. Arithmetic APIs update this
//! metadata for you, while maintenance helpers let you compact or resize owned
//! buffers without violating those invariants.
//!
//! Safe add/sub operations return K-normalized ciphertexts. Their
//! unnormalized variants live on [`api::CKKSAddOps`] and [`api::CKKSSubOps`]
//! and write into an [`layouts::UnnormalizedCKKSCiphertext`] for callers who
//! want to fuse several linear steps before normalizing explicitly. Limb
//! digits in that wrapper may hold un-propagated carries (wider than `base2k`
//! bits), so passing it to any DFT-domain primitive (keyswitching,
//! convolution, automorphisms) would produce incorrect decryptions. The
//! wrapper does not implement [`GLWEToBackendRef`] or [`GLWEToBackendMut`],
//! making such misuse a compile error. Call
//! [`layouts::UnnormalizedCKKSCiphertext::normalize`] before the next
//! keyswitching or convolution step.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`encoding`] | CKKS encoders/decoders, including slot-wise real/imaginary packing |
//! | [`layouts`] | CKKS ciphertext/plaintext wrappers and metadata-aware allocation helpers |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | [`api::CKKSBootstrappingOps`] | The CKKS bootstrapping pipeline: its one native primitive ModUp (modulus raise), plus CoeffsToSlots / SlotsToCoeffs and EvalMod re-exported as supertraits ([`api::DFTOps`] / [`api::CKKSEvalModOps`]); parameterized by [`layouts::BootstrappingPlan`] |

use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::Backend;

pub mod api;
pub(crate) mod cosine;
pub mod default;
pub(crate) mod delegates;

/// Re-exports for use inside this crate's exported macros (e.g.
/// [`impl_ckks_dft_defaults`]), so an invoking backend crate does not need
/// `anyhow` as a direct dependency. Not part of the public API.
#[doc(hidden)]
pub mod __macro_reexports {
    pub use anyhow;
}
pub mod encoding;
mod error;
pub mod layouts;
pub mod leveled;
pub mod oep;
pub mod polynomial;
pub mod power_basis;
#[cfg(not(all(feature = "libquadmath", target_arch = "x86_64")))]
pub mod scalar;
pub mod test_suite;
pub use error::CKKSCompositionError;
pub(crate) use error::{
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget, ensure_base2k_match,
    ensure_plaintext_alignment, ensure_plaintext_coeff_in_range, ensure_plaintext_degree_match,
};
#[cfg(all(feature = "libquadmath", target_arch = "x86_64"))]
pub use f128::f128 as Quad;
/// Quad-precision (IEEE 754 binary128) CKKS scalar.
///
/// The portable [`scalar::Quad`] newtype over the primitive `f128` by default;
/// the libquadmath-backed `f128::f128` under the `libquadmath` feature on
/// x86_64 (faster transcendentals for on-the-fly FFT-table builds).
#[cfg(not(all(feature = "libquadmath", target_arch = "x86_64")))]
pub use scalar::Quad;

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
/// `log_delta` is the scaling precision of the encoded value. The remaining
/// homomorphic headroom (`log_budget`) is *not* stored here: it is derived from
/// the wrapped GLWE's torus width `k` as `log_budget = k - log_delta`.
pub struct CKKSMeta {
    /// Base 2 logarithm of the decimal precision.
    pub log_delta: usize,
    /// Sparse-packing factor: `log2` of the coefficient gap (equivalently, of the
    /// slot replication). `0` is dense / full packing (`N/2` slots). For
    /// `log_sparsity = s` the message polynomial is sparse — `M(X^{2^s})` — and
    /// carries `(N/2) >> s` distinct slots, each replicated `2^s` times, i.e. a
    /// coefficient gap of `2^s`.
    pub log_sparsity: usize,
}

/// Common metadata accessors for CKKS ciphertext and plaintext containers.
///
/// This trait exposes the semantic precision of a value independently from the
/// raw limb storage used by the underlying torus representation. `log_budget` is
/// derived from the container's torus width `k` (from the wrapped GLWE) and
/// `log_delta`, so it is only available on containers, not on a bare [`CKKSMeta`].
pub trait CKKSInfos {
    /// Returns the complete metadata pair.
    fn meta(&self) -> CKKSMeta;

    /// Returns the base-2 logarithm of the encoded decimal scaling factor.
    fn log_delta(&self) -> usize;

    /// Returns the base-2 logarithm of the remaining homomorphic capacity.
    fn log_budget(&self) -> usize;

    /// Returns the sparse-packing factor (`log2` of the coefficient gap / slot
    /// replication); `0` is dense. See [`CKKSMeta::log_sparsity`].
    fn log_sparsity(&self) -> usize {
        self.meta().log_sparsity
    }
}

/// Mutable CKKS metadata access for ciphertext/plaintext containers.
pub trait SetCKKSInfos: CKKSInfos {
    /// Replaces the semantic CKKS metadata (`log_delta`, `log_sparsity`). Does not
    /// touch the wrapped GLWE's torus width `k`, so `log_budget` is re-derived
    /// against the (unchanged) `k`. Use [`Self::set_log_delta`] to relabel the
    /// scale while preserving `log_budget`.
    fn set_meta(&mut self, meta: CKKSMeta);

    /// Sets the wrapped GLWE's torus width `k` (the total `log_delta + log_budget`).
    fn set_k(&mut self, k: TorusPrecision);

    /// Updates only the base-2 logarithm of the encoded scaling factor, preserving
    /// `log_budget` by shifting the torus width `k` accordingly.
    fn set_log_delta(&mut self, log_delta: usize) {
        let log_budget = self.log_budget();
        let mut meta = self.meta();
        meta.log_delta = log_delta;
        self.set_meta(meta);
        self.set_k((log_budget + log_delta).into());
    }

    /// Updates only the base-2 logarithm of the remaining homomorphic budget by
    /// setting the torus width `k = log_budget + log_delta`.
    fn set_log_budget(&mut self, log_budget: usize) {
        self.set_k((log_budget + self.log_delta()).into());
    }

    /// Updates only the sparse-packing factor. See [`CKKSMeta::log_sparsity`].
    fn set_log_sparsity(&mut self, log_sparsity: usize) {
        let mut meta = self.meta();
        meta.log_sparsity = log_sparsity;
        self.set_meta(meta);
    }
}

/// Allocation / precision spec for a CKKS value.
///
/// Bundles a core [`GLWELayout`] — which carries `n`, `base2k`, the torus width
/// `k`, and `rank` — with the [`CKKSMeta`] (`log_delta`, `log_sparsity`). The
/// budget is derived as `log_budget = k - log_delta`, so `k` lives in the GLWE
/// layout exactly as it does on a wrapped ciphertext/plaintext.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CKKSLayout {
    pub glwe_layout: GLWELayout,
    pub meta: CKKSMeta,
}

impl Default for CKKSLayout {
    fn default() -> Self {
        Self {
            glwe_layout: GLWELayout {
                n: Degree(0),
                base2k: Base2K(0),
                k: TorusPrecision(0),
                rank: Rank(1),
            },
            meta: CKKSMeta::default(),
        }
    }
}

impl LWEInfos for CKKSLayout {
    fn n(&self) -> Degree {
        self.glwe_layout.n()
    }

    fn base2k(&self) -> Base2K {
        self.glwe_layout.base2k()
    }

    fn max_size(&self) -> usize {
        self.glwe_layout.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.glwe_layout.k()
    }
}

impl GLWEInfos for CKKSLayout {
    fn rank(&self) -> Rank {
        self.glwe_layout.rank()
    }
}

impl CKKSInfos for CKKSLayout {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta
    }

    fn log_budget(&self) -> usize {
        self.glwe_layout.k().as_usize().saturating_sub(self.meta.log_delta)
    }
}

pub(crate) fn ckks_offset_binary<R, A, B>(res: &R, a: &A, b: &B) -> usize
where
    R: LWEInfos + CKKSInfos + ?Sized,
    A: LWEInfos + CKKSInfos + ?Sized,
    B: LWEInfos + CKKSInfos + ?Sized,
{
    a.k().min(b.k()).as_usize().saturating_sub(res.max_k().as_usize())
}

pub(crate) fn ckks_offset_unary<R, A>(res: &R, a: &A) -> usize
where
    R: LWEInfos + CKKSInfos + ?Sized,
    A: LWEInfos + CKKSInfos + ?Sized,
{
    a.k().as_usize().saturating_sub(res.max_k().as_usize())
}
