//! Homomorphic DFT parameters (CoeffsToSlots / SlotsToCoeffs).
//!
//! Scheme-level description of a factorized homomorphic (I)DFT. The factor
//! matrices themselves are generated from this literal by the `default::dft`
//! module; this file holds the (backend-free) parameter struct, its enums, and
//! the prepared [`DFTMatrix`] that carries the encoded/prepared factor operands.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the full
//! design.

use core::marker::PhantomData;
use std::collections::BTreeSet;

use poulpy_core::{
    LinearTransformationPrepared,
    layouts::{LinearTransformation, LinearTransformationLayout, LinearTransformationStrategy},
};
use poulpy_hal::layouts::{Backend, galois_element};

use crate::{CKKSLayout, CKKSMeta, layouts::CKKSPlaintext};

/// Distinguishes the two homomorphic transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DFTType {
    /// Homomorphic encoding (IDFT), a.k.a. `CoeffsToSlots`.
    Encode,
    /// Homomorphic decoding (DFT), a.k.a. `SlotsToCoeffs`.
    Decode,
}

/// Input/output format of the homomorphic DFT.
///
/// `Standard` is the regular complex transform; the other two split the real and
/// imaginary parts (needed for bootstrapping).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DFTOutputFormat {
    /// Regular DFT: `[a+bi, c+di] -> DFT([a+bi, c+di])`.
    Standard,
    /// `Encode` returns the real and imaginary parts as two separate real
    /// vectors: `[a+bi, c+di] -> DFT([a, c]) and DFT([b, d])`.
    SplitRealAndImag,
    /// Like [`Self::SplitRealAndImag`] but, when sparsely packed (≤ N/4 slots),
    /// repacks the real part into the left N/2 real slots and the imaginary part into
    /// the right N/2 real slots. `Encode` and `Decode` must agree on this format:
    /// `[a+bi, c+di, a+bi, c+di] -> DFT([a, c, b, d])`.
    RepackImagAsReal,
}

/// Parameters describing a factorized homomorphic (I)DFT.
///
/// `factorization_depth` is the factorization schedule: one entry per factor
/// matrix, giving how many radix-2 FFT layers that matrix merges. The total
/// number of layers — and hence `log_slots` — is its **sum**, and the number of
/// factor matrices is its **length**. The schedule is free: pick the granularity
/// you want (`vec![1; log_slots]` = one layer per matrix, no merging;
/// `vec![log_slots]` = a single fully-merged matrix; anything in between). A
/// single uniform per-factor scale is used (see `docs/ckks_dft.md` §6).
///
/// **Convention: the schedule is in evaluation order** — the factor matrices are
/// applied to the ciphertext left-to-right, so `factorization_depth[0]` is the
/// first matrix evaluated and `factorization_depth[len-1]` the last. This holds
/// for both `Encode` and `Decode`; the generator does **no** implicit reordering
/// by `kind`. Because `Decode` is the inverse of `Encode`, a `Decode` plan that
/// undoes an `Encode` plan with schedule `s` uses the **reversed** schedule
/// (`s` read right-to-left) — that reversal is the caller's, not the library's.
/// (Symmetric schedules such as `vec![1; log_slots]` are their own reverse, so
/// the same schedule round-trips.)
#[derive(Clone, Debug)]
pub struct DFTPlan {
    /// Encode (IDFT) or Decode (DFT).
    pub kind: DFTType,
    /// Factorization schedule in **evaluation order**: `factorization_depth[i]`
    /// is the number of FFT layers merged into the `i`-th matrix applied to the
    /// ciphertext. `log_slots` is the sum (see [`Self::log_slots`]); the factor
    /// count is the length. Every entry must be `>= 1`. See the struct docs for
    /// the Encode/Decode reverse relationship.
    pub factorization_depth: Vec<usize>,
    /// BSGS giant-step width for each factor matrix, parallel to
    /// `factorization_depth` (same length). `factor_giant_steps[i]` is the width
    /// the `i`-th factor's diagonals are decomposed with; `1` means the direct
    /// schedule (one giant rotation per diagonal, no baby sharing).
    ///
    /// The schedule choice is the caller's: the library applies no implicit
    /// optimum, since the cost-optimal width depends on the backend. To compute a
    /// heuristic width, generate the factor's diagonal indexes and pass them to
    /// [`optimal_bsgs_giant_step`](poulpy_core::layouts::optimal_bsgs_giant_step).
    /// Each width is interpreted modulo that factor's own slot count (which is
    /// `2·slots` for the sparse-repack factors).
    pub giant_steps: Vec<usize>,
    /// Post-processing format. Default [`DFTOutputFormat::Standard`].
    ///
    /// On a *resolved* plan (one stored inside a [`DFTMatrix`]) this is
    /// canonical: a dense (non-sparse) `RepackImagAsReal` request is normalized
    /// to [`DFTOutputFormat::SplitRealAndImag`] by the constructor, so
    /// `RepackImagAsReal` here always means the sparse repack (see
    /// [`DFTMatrix::is_sparse`]).
    pub format: DFTOutputFormat,
    /// Constant the matrix is multiplied by. Default `1.0`.
    pub scaling: Option<f64>,
    /// If true, applies the transform bit-reversed (and expects bit-reversed
    /// inputs). Default false.
    pub bit_reversed: bool,
    /// `log_budget` bits each factor consumes (the per-factor plaintext
    /// `log_delta`). Meaningless on an input literal; the constructor fills it
    /// from `factor_meta` on the resolved plan stored in a [`DFTMatrix`].
    pub meta: CKKSLayout,
}

impl DFTPlan {
    /// `log2` of the number of complex slots the transform acts on: the sum of
    /// the factorization schedule (total FFT layers).
    pub fn log_slots(&self) -> usize {
        self.factorization_depth.iter().sum()
    }

    /// Number of factor matrices (the schedule length).
    pub fn num_factors(&self) -> usize {
        self.factorization_depth.len()
    }

    /// Validates the basic shape invariant shared by generation and evaluation:
    /// at least one factor, every factor merges at least one FFT layer, and the
    /// per-factor BSGS widths line up with the factorization schedule.
    pub fn check(&self) -> Result<(), String> {
        if self.factorization_depth.is_empty() {
            return Err("invalid DFTPlan: empty factorization_depth (no factor matrices)".to_string());
        }
        if self.factorization_depth.contains(&0) {
            return Err(format!(
                "invalid DFTPlan: factorization_depth has a zero-layer factor: {:?}",
                self.factorization_depth
            ));
        }
        if self.giant_steps.len() != self.factorization_depth.len() {
            return Err(format!(
                "invalid DFTPlan: factor_giant_steps (len {}) must match factorization_depth (len {})",
                self.giant_steps.len(),
                self.factorization_depth.len()
            ));
        }
        if self.giant_steps.contains(&0) {
            return Err(format!(
                "invalid DFTPlan: factor_giant_steps has a zero-width factor (use 1 for the direct schedule): {:?}",
                self.giant_steps
            ));
        }
        Ok(())
    }

    pub fn consumed_bits(&self) -> usize {
        self.num_factors() * self.meta.meta.log_delta
    }

    /// Whether this plan resolves to the sparse `RepackImagAsReal` path on a ring
    /// of degree `2^log_n` (a repack format with `log_slots < log_n − 1`). A dense
    /// `RepackImagAsReal` collapses to `SplitRealAndImag`, so it is **not** sparse.
    /// Matches [`DFTMatrix::is_sparse`] on the compiled matrix.
    pub fn is_sparse_repack(&self, log_n: usize) -> bool {
        self.format == DFTOutputFormat::RepackImagAsReal && self.log_slots() < log_n.saturating_sub(1)
    }

    /// Non-zero diagonal indexes of each factor matrix, in evaluation order
    /// (`diagonal_indexes(..)[i]` sorted ascending), **derived without generating
    /// any diagonal value**.
    ///
    /// The radix-2 butterfly merge spreads every diagonal `d` of a factor to
    /// `{d, (d+rot), (d−rot)}` (modulo the factor's slot count) for each merged
    /// FFT layer, where `rot` depends only on the layer level, `kind` and
    /// `bit_reversed` — never on the twiddle values. This replays exactly that
    /// index arithmetic, so it mirrors the diagonals the `default::dft` generator
    /// produces. `log_n` (ring degree exponent) is needed only to decide the
    /// sparse-repack path (which prepends the `{0, slots}` repack diagonals and
    /// widens the first `Decode` factor to `2·slots`).
    pub fn diagonal_indexes(&self, log_n: usize) -> Vec<Vec<i64>> {
        self.check().expect("invalid DFTPlan");

        let log_slots = self.log_slots();
        let slots = 1i64 << log_slots;
        let sparse = self.is_sparse_repack(log_n);
        // `rot` uses the absolute level for Encode-forward / Decode-bit-reversed,
        // and `log_slots − level` otherwise (cf. `default::dft::rot_uses_level`).
        let uses_level =
            (self.kind == DFTType::Encode && !self.bit_reversed) || (self.kind == DFTType::Decode && self.bit_reversed);

        let mut out = Vec::with_capacity(self.num_factors());
        let mut fft_level = log_slots as i64;
        for (i, &m) in self.factorization_depth.iter().enumerate() {
            // Only the sparse-repack first Decode factor starts from the repack
            // matrix ({0, slots}) and wraps its rotations modulo 2·slots.
            let repack_first = sparse && self.kind == DFTType::Decode && i == 0;
            let merge_n = if repack_first { slots << 1 } else { slots };
            let mask = merge_n - 1;

            let mut set: BTreeSet<i64> = if repack_first {
                BTreeSet::from([0, slots])
            } else {
                BTreeSet::from([0])
            };
            let mut level = fft_level;
            for _ in 0..m {
                let rot = if uses_level {
                    (1i64 << (level - 1)) & mask
                } else {
                    (1i64 << (log_slots as i64 - level)) & mask
                };
                let mut next: BTreeSet<i64> = BTreeSet::new();
                for &d in &set {
                    next.insert(d);
                    next.insert((d + rot) & mask);
                    next.insert((d - rot) & mask);
                }
                set = next;
                level -= 1;
            }
            out.push(set.into_iter().collect());
            fft_level -= m as i64;
        }
        out
    }

    /// Number of non-zero diagonals of each factor matrix, in evaluation order
    /// (the per-factor lengths of [`diagonal_indexes`](Self::diagonal_indexes)).
    pub fn num_diagonals(&self, log_n: usize) -> Vec<usize> {
        self.diagonal_indexes(log_n).iter().map(Vec::len).collect()
    }

    /// Distinct Galois elements whose automorphism keys evaluating this transform
    /// requires, on a ring of degree `2^log_n` with the given `cyclotomic_order`.
    ///
    /// Derived from the per-factor [`diagonal_indexes`](Self::diagonal_indexes) and
    /// per-factor BSGS `giant_steps` (plus the `slots` repack rotation on the
    /// sparse path), without generating any diagonal. Equals
    /// [`DFTMatrix::galois_elements`] on the compiled matrix. The conjugation key
    /// used by the split forward transform is separate and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let sparse = self.is_sparse_repack(log_n);
        let slots = 1i64 << self.log_slots();
        // The factor diagonals (and thus the BSGS schedule) live in the `2·slots`
        // working ring on the sparse path, else in `slots`.
        let slot_count = if sparse { (slots << 1) as usize } else { slots as usize };

        let mut set: BTreeSet<i64> = BTreeSet::new();
        for (indexes, &giant_step) in self.diagonal_indexes(log_n).into_iter().zip(&self.giant_steps) {
            let layout = LinearTransformationLayout {
                indexes,
                slots: slot_count,
                strategy: LinearTransformationStrategy::Bsgs { giant_step },
            };
            set.extend(layout.galois_elements(cyclotomic_order));
        }
        if sparse {
            set.insert(galois_element(slots, cyclotomic_order));
        }
        // Defensive: the identity automorphism is never key-switched.
        set.remove(&galois_element(0, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// The factor operands shared by every [`DFTMatrix`] variant: the per-factor
/// right operands (one [`DFTPlan`] factor each), in evaluation order, plus the
/// resolved plan they were generated from.
///
/// Generic over the factor representation `R`: an unprepared
/// [`LinearTransformation`] whose plaintext diagonals are materialized on the fly
/// at eval time (the default — host-resident, streamed) or a prepared
/// [`LinearTransformationPrepared`] keeping the convolution-domain diagonals
/// resident ([`DFTMatrixPrepared`]). See [`DFTMatrix`].
pub struct DFTMatrixFactors<BE: Backend, R = LinearTransformation<CKKSPlaintext<<BE as Backend>::OwnedBuf>>> {
    /// The resolved parameters this matrix was generated from (canonical
    /// `format`, populated `factor_log_delta`).
    pub plan: DFTPlan,
    /// Per-factor right operands, one per factor matrix, in evaluation order.
    pub(crate) factors: Vec<R>,
    _backend: core::marker::PhantomData<BE>,
}

impl<BE: Backend, R> DFTMatrixFactors<BE, R> {
    pub(crate) fn new(plan: DFTPlan, factors: Vec<R>) -> Self {
        Self {
            plan,
            factors,
            _backend: core::marker::PhantomData,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Transform-direction marker: homomorphic encoding (`CoeffsToSlots`, the IDFT).
pub struct Encode;
/// Transform-direction marker: homomorphic decoding (`SlotsToCoeffs`, the DFT).
pub struct Decode;
impl sealed::Sealed for Encode {}
impl sealed::Sealed for Decode {}

/// Compile-time transform direction carried in a [`DFTMatrix`] type parameter.
/// Implemented by [`Encode`] / [`Decode`]; sealed.
pub trait DftDirection: sealed::Sealed {
    /// The runtime [`DFTType`] this marker stands for.
    const KIND: DFTType;
}
impl DftDirection for Encode {
    const KIND: DFTType = DFTType::Encode;
}
impl DftDirection for Decode {
    const KIND: DFTType = DFTType::Decode;
}

/// Output-format marker: regular complex transform ([`DFTOutputFormat::Standard`]).
pub struct Standard;
/// Output-format marker: real/imag split ([`DFTOutputFormat::SplitRealAndImag`]).
pub struct Split;
/// Output-format marker: sparse `RepackImagAsReal` (imag repacked into the right
/// half).
pub struct Repack;
impl sealed::Sealed for Standard {}
impl sealed::Sealed for Split {}
impl sealed::Sealed for Repack {}

/// Compile-time output format carried in a [`DFTMatrix`] type parameter.
/// Implemented by [`Standard`] / [`Split`] / [`Repack`]; sealed.
pub trait DftFormat: sealed::Sealed {
    /// The runtime [`DFTOutputFormat`] this marker stands for.
    const FORMAT: DFTOutputFormat;
}
impl DftFormat for Standard {
    const FORMAT: DFTOutputFormat = DFTOutputFormat::Standard;
}
impl DftFormat for Split {
    const FORMAT: DFTOutputFormat = DFTOutputFormat::SplitRealAndImag;
}
impl DftFormat for Repack {
    const FORMAT: DFTOutputFormat = DFTOutputFormat::RepackImagAsReal;
}

/// A generated, ready-to-evaluate homomorphic (I)DFT, carrying its transform
/// **direction** `Dir` ([`Encode`]/[`Decode`]) and output **format** `Fmt`
/// ([`Standard`]/[`Split`]/[`Repack`]) as compile-time type-state, and generic
/// over the factor representation `R`.
///
/// Because direction and format live in the type, the evaluation entry points
/// require the exact matrix — e.g. [`ckks_coeffs_to_slots_repack`] only accepts a
/// `DFTMatrix<BE, Encode, Repack, R>` — so a direction/format mismatch is a
/// **compile error** rather than a runtime check. The single runtime resolution
/// (the dense-`RepackImagAsReal`≡`Split` rule, which depends on `log_slots` vs
/// `log_n`) happens once, when [`ckks_new_dft_matrix`] establishes the markers
/// (it errors if `Repack` is requested for dense parameters).
///
/// `R` selects how each factor's RHS is stored, trading memory for compute: the
/// default keeps unprepared plaintext diagonals ([`LinearTransformation`]) and
/// materializes them per factor at eval time (minimal resident memory); the
/// prepared alias [`DFTMatrixPrepared`] keeps the convolution-domain diagonals
/// resident. Build via [`ckks_new_dft_matrix`], then optionally
/// [`ckks_prepare_dft_matrix`]. The required Galois keys are reported by
/// `galois_elements`.
///
/// [`ckks_new_dft_matrix`]: crate::default::dft::ckks_new_dft_matrix
/// [`ckks_prepare_dft_matrix`]: crate::default::dft::ckks_prepare_dft_matrix
/// [`ckks_coeffs_to_slots_repack`]: crate::default::dft::ckks_coeffs_to_slots_repack
pub struct DFTMatrix<BE: Backend, Dir, Fmt, R = LinearTransformation<CKKSPlaintext<<BE as Backend>::OwnedBuf>>> {
    pub(crate) inner: DFTMatrixFactors<BE, R>,
    _marker: PhantomData<(Dir, Fmt)>,
}

/// Prepared (resident-RHS) [`DFTMatrix`]: each factor's diagonals are kept in the
/// convolution domain ([`LinearTransformationPrepared`]) rather than materialized
/// per factor, trading resident memory for faster repeated evaluation. Preserves
/// the `Dir`/`Fmt` type-state of the matrix it was prepared from. Obtained by
/// preparing a [`DFTMatrix`] via
/// [`ckks_prepare_dft_matrix`](crate::default::dft::ckks_prepare_dft_matrix).
pub type DFTMatrixPrepared<BE, Dir, Fmt> = DFTMatrix<BE, Dir, Fmt, LinearTransformationPrepared<BE>>;

impl<BE: Backend, Dir, Fmt, R> DFTMatrix<BE, Dir, Fmt, R> {
    /// Wraps factor operands into the typed matrix. Internal: the caller asserts
    /// the `Dir`/`Fmt` markers describe the resolved plan (see
    /// `ckks_new_dft_matrix`, which validates this at construction).
    pub(crate) fn from_factors(inner: DFTMatrixFactors<BE, R>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// The factor operands.
    pub(crate) fn inner(&self) -> &DFTMatrixFactors<BE, R> {
        &self.inner
    }

    /// The resolved plan (canonical `format`, populated `factor_log_delta`).
    pub fn plan(&self) -> &DFTPlan {
        &self.inner.plan
    }

    /// The per-factor right operands, in evaluation order.
    pub(crate) fn factor_operands(&self) -> &[R] {
        &self.inner.factors
    }

    /// Number of factor matrices (one linear transformation each).
    pub fn num_factors(&self) -> usize {
        self.inner.factors.len()
    }

    /// `log_budget` bits consumed per factor (the per-factor plaintext `log_delta`).
    pub fn meta(&self) -> CKKSMeta {
        self.inner.plan.meta.meta
    }

    /// Total `log_budget` bits the whole transform consumes: `num_factors ×
    /// factor_log_delta`. The input ciphertext must have at least this much.
    pub fn consumed_bits(&self) -> usize {
        self.plan().consumed_bits()
    }
}

impl<BE: Backend, Dir, Fmt: DftFormat, R> DFTMatrix<BE, Dir, Fmt, R> {
    /// Whether this is the sparse `RepackImagAsReal` path (needs the `slots`
    /// repack rotation and updates `log_sparsity`). A compile-time property of the
    /// `Fmt` type-state.
    pub fn is_sparse(&self) -> bool {
        Fmt::FORMAT == DFTOutputFormat::RepackImagAsReal
    }
}
