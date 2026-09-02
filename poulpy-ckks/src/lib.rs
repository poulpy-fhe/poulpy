#![feature(f128)]
#![deny(rustdoc::broken_intra_doc_links)]
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
//! [`CKKSMeta`] also records the [`SlotsKind`] of a value: whether its slots are
//! known to be real, or may carry an imaginary part. Operations compose that
//! claim, keeping `Real` only when every operand is real, so a caller can state
//! it once at encoding time and have the pipelines specialize on it.
//!
//! Together they define the semantic torus width of a value:
//! `k() = log_delta + log_budget`.
//! Storage is rounded up to the next multiple of `base2k`, so the allocated
//! capacity `max_k()` may exceed the effective width `k()`. Arithmetic APIs
//! update this metadata for you; buffers always stay at their allocated
//! width, and allocating a destination at exactly the `k` you want is how
//! results are narrowed.
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
//! | [`approximation`] | Reusable minimax fitting, precision/depth selection, composite sign generation, and prepared interval-mapped polynomial evaluation |
//! | [`encoding`] | CKKS encoders/decoders, including slot-wise real/imaginary packing |
//! | [`layouts`] | CKKS ciphertext/plaintext wrappers and metadata-aware allocation helpers |
//! | [`presets`] | Ready-to-use parameter sets |
//! | [`api`] | The public op traits: leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, rescale, and scratch sizing |
//! | [`api::CKKSBootstrappingOps`] | The CKKS bootstrapping pipeline: its one native primitive ModUp (modulus raise), plus CoeffsToSlots / SlotsToCoeffs and EvalMod re-exported as supertraits ([`api::CKKSDFTOps`] / [`api::CKKSEvalModOps`]); parameterized by [`layouts::BootstrappingPlan`] |
//! | [`api::CKKSPaCoOps`] | PaCo bootstrapping without ModUp or EvalMod; parameterized by [`layouts::PaCoPlan`] and a compiled [`layouts::PaCoContext`] |

use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::Backend;

pub mod api;
pub mod approximation;
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
mod eval_lut;
pub mod layouts;
/// One-stop imports for the common CKKS path.
///
/// `use poulpy_ckks::prelude::*;` brings in the op traits (add/sub/mul/…,
/// encrypt/decrypt, scratch sizing), the ciphertext/plaintext containers with
/// their allocation helpers, and the metadata types — everything the
/// encode → encrypt → evaluate → decrypt → decode loop needs from this crate.
/// Key material and the `Module`/scratch machinery come from `poulpy-core` and
/// `poulpy-hal`.
pub mod prelude {
    pub use crate::api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSApproximationOps, CKKSConjugateOps, CKKSCopyOps, CKKSDecryptOps, CKKSEncodingHostOps,
        CKKSEncodingOps, CKKSEncryptOps, CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSPlaintextVecOps, CKKSPow2Ops, CKKSRotateOps,
        CKKSSubOps,
    };
    pub use crate::layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, PolynomialApproximation, UnnormalizedCKKSCiphertext,
    };
    pub use crate::{
        CKKSCompositionError, CKKSError, CKKSInfos, CKKSLayout, CKKSMeta, CKKSResult, CoeffsMeta, Quad, SetCKKSInfos, SlotsKind,
    };
}
pub mod oep;
pub mod polynomial;
pub mod power_basis;
pub mod presets;
pub mod scalar;
#[cfg(feature = "test-utils")]
pub mod test_suite;
pub use error::{CKKSCompositionError, CKKSError, CKKSResult};
pub(crate) use error::{
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget, ckks_bail, ckks_ensure, ensure_base2k_match,
    ensure_plaintext_alignment, ensure_plaintext_coeff_in_range, ensure_plaintext_degree_match,
};
/// Quad-precision (IEEE 754 binary128) CKKS scalar.
///
/// Always the portable [`scalar::Quad`] newtype over the primitive `f128`, in
/// every configuration. Under the `libquadmath` feature on non-Apple x86_64
/// only its libm-backed math is routed through libquadmath (faster for
/// on-the-fly FFT-table builds); storage, codecs, and exact arithmetic are
/// unchanged, so the type — and its `bytemuck::Pod` encoding — is identical
/// across features.
pub use scalar::Quad;

/// Backend-compatible shared CKKS plaintext storage.
pub trait CKKSPlaintextToBackendRef<BE: Backend>: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos {}

impl<BE: Backend, T> CKKSPlaintextToBackendRef<BE> for T where T: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos {}

/// Backend-compatible mutable CKKS plaintext storage.
///
/// Implemented by owned plaintexts and scratch-backed plaintext views alike.
pub trait CKKSPlaintextToBackendMut<BE: Backend>: CKKSPlaintextToBackendRef<BE> + GLWEToBackendMut<BE> {}

impl<BE: Backend, T> CKKSPlaintextToBackendMut<BE> for T where T: CKKSPlaintextToBackendRef<BE> + GLWEToBackendMut<BE> {}

/// Marker bound for CKKS ciphertext type parameters.
///
/// Combines [`GLWEInfos`] (which already implies [`poulpy_core::layouts::LWEInfos`])
/// with [`CKKSInfos`] to collapse the repeated `GLWEInfos + CKKSInfos` pair found
/// throughout the API into a single, named constraint.
pub trait CKKSCtBounds: GLWEInfos + CKKSInfos {}

impl<T: GLWEInfos + CKKSInfos> CKKSCtBounds for T {}

/// Which subfield the encoded slots are known to live in.
///
/// The reals are a subring of the complexes, so the two variants are ordered
/// claims rather than exclusive tags: [`SlotsKind::Real`] is the stronger one,
/// [`SlotsKind::Complex`] is always sound. Operations compose the claim with
/// [`SlotsKind::join`], which keeps `Real` only when every operand is `Real`.
/// `Complex` is the default, so a value that never states its kind is never
/// mistaken for a real one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SlotsKind {
    /// Every slot has a zero imaginary part.
    Real,
    /// Slots may carry a nonzero imaginary part.
    #[default]
    Complex,
}

impl SlotsKind {
    /// Kind of a value built from two operands: `Real` only when both are.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Real, Self::Real) => Self::Real,
            _ => Self::Complex,
        }
    }

    /// Whether the slots are known to be real.
    pub fn is_real(self) -> bool {
        self == Self::Real
    }
}

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
    /// Subfield the slots are known to live in. See [`SlotsKind`].
    pub slots: SlotsKind,
}

/// Common metadata accessors for CKKS ciphertext and plaintext containers.
///
/// This trait exposes the semantic precision of a value independently from the
/// raw limb storage used by the underlying torus representation. `log_budget` is
/// derived from the container's torus width `k` (from the wrapped GLWE) and
/// `log_delta`, so it is only available on containers, not on a bare [`CKKSMeta`].
pub trait CKKSInfos: LWEInfos {
    /// Returns the complete metadata pair.
    fn meta(&self) -> CKKSMeta;

    /// Returns the base-2 logarithm of the encoded decimal scaling factor.
    fn log_delta(&self) -> usize {
        self.meta().log_delta
    }

    /// Returns the base-2 logarithm of the remaining homomorphic capacity,
    /// derived from the container's torus width: `k − log_delta`.
    fn log_budget(&self) -> usize {
        self.k().as_usize().saturating_sub(self.log_delta())
    }

    /// Returns the sparse-packing factor (`log2` of the coefficient gap / slot
    /// replication); `0` is dense. See [`CKKSMeta::log_sparsity`].
    fn log_sparsity(&self) -> usize {
        self.meta().log_sparsity
    }

    /// Returns the subfield the slots are known to live in. See [`SlotsKind`].
    fn slots(&self) -> SlotsKind {
        self.meta().slots
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

    /// Updates only the slot kind. See [`SlotsKind`].
    fn set_slots(&mut self, slots: SlotsKind) {
        let mut meta = self.meta();
        meta.slots = slots;
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

/// Coefficient metadata for plan-compiled operands: the DFT factor diagonals
/// ([`layouts::DFTPlan`]), the EvalMod polynomial coefficients
/// ([`layouts::EvalModPlan`]), and the BSGS polynomial encoders
/// ([`polynomial::EncodeBSGS`]).
///
/// The reduced form of a [`CKKSLayout`]: only the torus width `k` the operand
/// plaintexts are allocated with and the [`CKKSMeta`] they are stamped with.
/// Plans carry no ring or radix information — `n` follows the module and
/// `base2k` is passed explicitly at compile time — so there is nothing to fill
/// with placeholders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoeffsMeta {
    /// Torus width the operand plaintexts are allocated with.
    pub k: TorusPrecision,
    /// CKKS metadata (`log_delta`, `log_sparsity`) the operands are stamped with.
    pub meta: CKKSMeta,
}

impl CoeffsMeta {
    /// Dense (`log_sparsity = 0`) coefficient meta with
    /// `k = log_delta + log_budget`.
    pub fn from_delta_budget(log_delta: usize, log_budget: usize) -> Self {
        Self {
            k: (log_delta + log_budget).into(),
            meta: CKKSMeta {
                log_delta,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
        }
    }

    /// The operand encoding scale.
    pub fn log_delta(&self) -> usize {
        self.meta.log_delta
    }

    /// The operand headroom: `k − log_delta` (saturating).
    pub fn log_budget(&self) -> usize {
        usize::from(self.k).saturating_sub(self.meta.log_delta)
    }
}

/// Narrowing conversion: keeps `k` and the [`CKKSMeta`], drops the ring/radix
/// fields plans never consume.
impl From<CKKSLayout> for CoeffsMeta {
    fn from(layout: CKKSLayout) -> Self {
        Self {
            k: layout.glwe_layout.k,
            meta: layout.meta,
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
}

/// Bits a binary add/sub must shift its result down to fit `res`: the excess of
/// the **natural result width** — `min(log_delta) + min(log_budget)`, the meta
/// the operation stamps — over the destination's requested `res.k()`. Using the natural width (rather
/// than `min(a.k, b.k)`, which is `≥` it whenever deltas *and* budgets both
/// differ) charges exactly the budget the narrower destination forces and no
/// more; the larger-delta operand's truncated tail lies below the claimed
/// `min(log_delta)` precision. This mirrors the mul family's
/// `(res_log_budget + res_log_delta).saturating_sub(res_max_k)`.
pub(crate) fn ckks_offset_binary<R, A, B>(res: &R, a: &A, b: &B) -> usize
where
    R: CKKSInfos + ?Sized,
    A: CKKSInfos + ?Sized,
    B: CKKSInfos + ?Sized,
{
    let natural_k = a.log_delta().min(b.log_delta()) + a.log_budget().min(b.log_budget());
    natural_k.saturating_sub(res.k().as_usize())
}

pub(crate) fn ckks_offset_unary<R, A>(res: &R, a: &A) -> usize
where
    R: CKKSInfos + ?Sized,
    A: CKKSInfos + ?Sized,
{
    a.k().as_usize().saturating_sub(res.k().as_usize())
}

/// Shared unary-op preamble: aligns `src` into `dst` (left shift by
/// `offset + extra_shift`) and stamps `src`'s metadata with the budget charged
/// by `offset + extra_charge`. Validates **before** mutating, so on `Err`
/// (insufficient budget) `dst` is untouched. Returns the computed offset.
///
/// This is the single implementation of the "shift + stamp" sequence the
/// copy/pow2/add-pt/sub-pt into-ops previously hand-rolled (H1 in the 2026-07
/// review was a drift bug in exactly this preamble).
pub(crate) fn ckks_shift_stamp_unary<BE, M, Dst, Src>(
    module: &M,
    op: &'static str,
    dst: &mut Dst,
    src: &Src,
    extra_shift: usize,
    extra_charge: usize,
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: poulpy_core::GLWEShift<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos,
{
    let offset = ckks_offset_unary(dst, src);
    let log_budget = checked_log_budget_sub(op, src.log_budget(), offset + extra_charge)?;
    module.glwe_lsh(dst, src, offset + extra_shift, scratch);
    dst.set_meta(src.meta());
    dst.set_log_budget(log_budget);
    Ok(())
}

#[cfg(test)]
mod slots_kind_tests {
    use super::SlotsKind::{self, Complex, Real};

    #[test]
    fn join_keeps_real_only_when_both_operands_are_real() {
        assert_eq!(Real.join(Real), Real);
        assert_eq!(Real.join(Complex), Complex);
        assert_eq!(Complex.join(Real), Complex);
        assert_eq!(Complex.join(Complex), Complex);
    }

    #[test]
    fn unstated_kind_is_complex() {
        assert_eq!(SlotsKind::default(), Complex);
        assert!(!SlotsKind::default().is_real());
    }
}

#[cfg(test)]
mod offset_tests {
    use super::*;

    fn layout(log_delta: usize, log_budget: usize) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: poulpy_core::layouts::GLWELayout {
                n: Degree(0),
                base2k: Base2K(1),
                k: (log_delta + log_budget).into(),
                rank: Rank(1),
            },
            meta: CKKSMeta {
                log_delta,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
        }
    }

    /// The binary offset charges exactly the natural result width's excess over
    /// the destination — not `min(a.k, b.k)`, which over-charges when deltas
    /// and budgets both differ.
    #[test]
    fn offset_binary_charges_natural_result_width() {
        // Deltas AND budgets differ: natural width = 30 + 20 = 50, while
        // min(a.k, b.k) = 55 — the old formula charged 5 spurious bits.
        let a = layout(40, 20); // k = 60
        let b = layout(30, 25); // k = 55
        let res = layout(30, 20); // max_k = 50
        assert_eq!(ckks_offset_binary(&res, &a, &b), 0);

        // A genuinely narrower destination still charges the difference.
        let narrow = layout(30, 12); // max_k = 42
        assert_eq!(ckks_offset_binary(&narrow, &a, &b), 8);

        // Degenerate cases match the old formula: equal budgets…
        let c = layout(40, 25); // k = 65
        let d = layout(30, 25); // k = 55
        assert_eq!(ckks_offset_binary(&narrow, &c, &d), (30 + 25) - 42);
        // …and equal deltas.
        let e = layout(30, 30); // k = 60
        assert_eq!(ckks_offset_binary(&narrow, &d, &e), (30 + 25) - 42);
    }
}
