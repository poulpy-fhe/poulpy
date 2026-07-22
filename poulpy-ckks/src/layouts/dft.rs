//! Homomorphic DFT parameters (CoeffsToSlots / SlotsToCoeffs).
//!
//! Scheme-level description of a factorized homomorphic (I)DFT. The factor
//! matrices themselves are generated from this literal by the `default::dft`
//! module; this file holds the (backend-free) parameter struct, its enums, and
//! the prepared [`DFTMatrix`] that carries the encoded/prepared factor operands.
//! The homomorphic DFT is documented as a stage of the bootstrapping pipeline in
//! [`docs/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/bootstrapping.md).

use core::marker::PhantomData;
use std::collections::BTreeSet;

use poulpy_core::{
    LinearTransformationPrepared,
    layouts::{LinearTransformation, LinearTransformationLayout, LinearTransformationStrategy},
};
use poulpy_hal::layouts::{Backend, galois_element};

use crate::{CKKSMeta, CoeffsMeta, layouts::CKKSPlaintext};

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
/// Built with [`Self::new`], which validates the shape invariants once — a
/// `DFTPlan` that exists is always shape-valid, so the derived-index APIs
/// ([`Self::diagonal_indexes`], [`Self::galois_elements`]) are infallible.
///
/// The [`FactorSchedule`] gives one [`FactorStep`] per factor matrix: how many
/// radix-2 FFT layers that matrix merges (`depth`) and its BSGS giant-step
/// width. The total number of layers — and hence `log_slots` — is the **sum of
/// depths**, and the number of factor matrices is the schedule **length**. The
/// schedule is free: pick the granularity you want (`vec![(1, g); log_slots]` =
/// one layer per matrix, no merging; `vec![(log_slots, g)]` = a single
/// fully-merged matrix; anything in between). A single uniform per-factor scale
/// is used (see the homomorphic-DFT scale accounting in
/// [`docs/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/bootstrapping.md)).
///
/// **Convention: the schedule is in evaluation order** — the factor matrices are
/// applied to the ciphertext left-to-right, so the first step is the
/// first matrix evaluated and the last step the last. This holds
/// for both `Encode` and `Decode`; the generator does **no** implicit reordering
/// by `kind`. Because `Decode` is the inverse of `Encode`, a `Decode` plan that
/// undoes an `Encode` plan with schedule `s` uses the **reversed** schedule
/// (`s` read right-to-left) — that reversal is the caller's, not the library's.
/// (Symmetric schedules such as `vec![(1, g); log_slots]` are their own reverse,
/// so the same schedule round-trips.)
#[derive(Clone, Debug)]
pub struct DFTPlan {
    /// Encode (IDFT) or Decode (DFT).
    pub(crate) kind: DFTType,
    /// Factorization schedule in **evaluation order** (see the struct docs).
    pub(crate) schedule: FactorSchedule,
    /// Post-processing format.
    ///
    /// On a *resolved* plan (one stored inside a [`DFTMatrix`]) this is
    /// canonical: a dense (non-sparse) `RepackImagAsReal` request is normalized
    /// to [`DFTOutputFormat::SplitRealAndImag`], so `RepackImagAsReal` there
    /// always means the sparse repack (see [`DFTMatrix::is_sparse`]).
    pub(crate) format: DFTOutputFormat,
    /// Constant the matrix is multiplied by; `None` = `1.0`. Set via
    /// [`Self::with_scaling`], which validates it.
    pub(crate) scaling: Option<f64>,
    /// If true, applies the transform bit-reversed (and expects bit-reversed
    /// inputs).
    pub(crate) bit_reversed: bool,
    /// Per-factor coefficient metadata: the torus width the factor plaintexts
    /// are allocated with, and the scale (`log_delta` — the `log_budget` bits
    /// each factor consumes) they are encoded at.
    pub(crate) coeffs_meta: CoeffsMeta,
}

/// Single source of truth for the radix-2 merge geometry, shared by the factor
/// generator (`crate::default::dft::matrices::merge_next_layer`) and the
/// value-free index replay ([`DFTPlan::diagonal_indexes`]): the rotation a
/// merged FFT layer applies. `level` counts down from `log_slots` (the layer
/// being merged into the factor); `mask` is the factor's slot-count mask
/// (`2·slots − 1` on the sparse-repack first `Decode` factor, else `slots − 1`).
///
/// Keeping this in one place is what guarantees key provisioning
/// ([`DFTPlan::galois_elements`]) covers exactly the diagonals the generator
/// produces — a drift between the two would silently under-provision keys.
pub(crate) fn dft_layer_rotation(kind: DFTType, bit_reversed: bool, level: usize, log_slots: usize, mask: i64) -> i64 {
    // Encode-forward and Decode-bit-reversed share the absolute-level branch.
    let uses_level = (kind == DFTType::Encode && !bit_reversed) || (kind == DFTType::Decode && bit_reversed);
    if uses_level {
        (1i64 << (level - 1)) & mask
    } else {
        (1i64 << (log_slots - level)) & mask
    }
}

/// The `{d, d+rot, d−rot}` (mod `mask + 1`) diagonal spread of one merged
/// butterfly layer — the index arithmetic paired with [`dft_layer_rotation`].
pub(crate) fn dft_layer_spread(d: i64, rot: i64, mask: i64) -> [i64; 3] {
    [d, (d + rot) & mask, (d - rot) & mask]
}

/// One factor of a factorization schedule: how many radix-2 layers the
/// factor matrix merges, and the BSGS giant-step width it is evaluated with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactorStep {
    /// Radix-2 layers merged into this factor; must be ≥ 1.
    pub depth: usize,
    /// BSGS giant-step width; `1` is the direct schedule. Must be ≥ 1.
    pub giant_step: usize,
}

/// The factorization schedule shared by every factorized homomorphic
/// transform plan ([`DFTPlan`] and the PaCo factor chains): one
/// [`FactorStep`] per factor matrix, in evaluation order.
///
/// Storing per-factor pairs (rather than two parallel vectors) makes the
/// "same length" invariant unrepresentable; the remaining shape invariants
/// are owned by [`Self::check`], through which both plan hierarchies
/// validate, so the clauses cannot drift the way two hand-copied validators
/// can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorSchedule {
    /// The per-factor steps, in evaluation order. Must be non-empty
    /// (checked).
    pub steps: Vec<FactorStep>,
}

impl FactorSchedule {
    /// Builds a schedule from parallel depth/giant-step lists, erroring (as
    /// `plan`) if their lengths differ. Shape invariants beyond the pairing
    /// are still checked separately via [`Self::check`].
    pub fn from_parts(factorization_depth: &[usize], giant_steps: &[usize], plan: &'static str) -> anyhow::Result<Self> {
        if giant_steps.len() != factorization_depth.len() {
            return Err(crate::CKKSCompositionError::InvalidPlan {
                plan,
                reason: format!(
                    "giant_steps (len {}) must match factorization_depth (len {})",
                    giant_steps.len(),
                    factorization_depth.len()
                ),
            }
            .into());
        }
        Ok(Self {
            steps: factorization_depth
                .iter()
                .zip(giant_steps)
                .map(|(&depth, &giant_step)| FactorStep { depth, giant_step })
                .collect(),
        })
    }

    /// Number of factor matrices (the schedule length).
    pub fn num_factors(&self) -> usize {
        self.steps.len()
    }

    /// Total merged layers: `log_slots` for a full DFT, the chain's unit
    /// count for the PaCo chains. Unchecked sum; use [`Self::check`] first
    /// when the schedule is caller-supplied.
    pub fn log_slots(&self) -> usize {
        self.steps.iter().map(|s| s.depth).sum()
    }

    /// The per-factor merged-layer counts, in evaluation order (collected;
    /// schedules are consulted on setup paths only).
    pub fn depths(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.depth).collect()
    }

    /// The per-factor BSGS giant-step widths, in evaluation order (collected;
    /// schedules are consulted on setup paths only).
    pub fn giant_steps(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.giant_step).collect()
    }

    /// Validates the shape invariants and returns the checked layer sum:
    /// at least one factor, no zero-layer factor, no zero giant step, and a
    /// non-overflowing sum.
    pub fn check(&self, plan: &'static str) -> anyhow::Result<usize> {
        let invalid = |reason: String| -> anyhow::Error { crate::CKKSCompositionError::InvalidPlan { plan, reason }.into() };
        if self.steps.is_empty() {
            return Err(invalid("empty factorization schedule (no factor matrices)".to_string()));
        }
        if self.steps.iter().any(|s| s.depth == 0) {
            return Err(invalid(format!(
                "factorization schedule has a zero-layer factor: {:?}",
                self.depths()
            )));
        }
        if self.steps.iter().any(|s| s.giant_step == 0) {
            return Err(invalid(format!(
                "schedule has a zero-width giant step (use 1 for the direct schedule): {:?}",
                self.giant_steps()
            )));
        }
        self.steps
            .iter()
            .try_fold(0usize, |sum, s| sum.checked_add(s.depth))
            .ok_or_else(|| invalid(format!("factorization schedule sum overflows usize: {:?}", self.depths())))
    }
}

/// `(depth, giant_step)` pairs, in evaluation order — the ergonomic literal
/// form: `vec![(2, 4), (2, 4), (3, 4)].into()`.
impl From<Vec<(usize, usize)>> for FactorSchedule {
    fn from(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            steps: pairs
                .into_iter()
                .map(|(depth, giant_step)| FactorStep { depth, giant_step })
                .collect(),
        }
    }
}

impl From<Vec<FactorStep>> for FactorSchedule {
    fn from(steps: Vec<FactorStep>) -> Self {
        Self { steps }
    }
}

impl DFTPlan {
    /// Validated constructor: the only way to obtain a `DFTPlan` outside the
    /// crate, so every plan in circulation is shape-valid (the derived-index
    /// APIs are therefore infallible).
    ///
    /// `schedule` accepts a [`FactorSchedule`] or `(depth, giant_step)` pairs in
    /// evaluation order (`vec![(2, 4), (3, 4)].into()` — see the struct docs for
    /// the ordering convention). `coeffs_meta` carries the per-factor
    /// coefficient metadata (its `log_delta` is the per-factor scale). `scaling`
    /// defaults to `1.0` and `bit_reversed` to `false`; adjust with
    /// [`Self::with_scaling`] / [`Self::with_bit_reversed`].
    ///
    /// Errors ([`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)) if the
    /// schedule fails [`FactorSchedule::check`] or the consumed-bit count
    /// overflows.
    pub fn new(
        kind: DFTType,
        schedule: impl Into<FactorSchedule>,
        format: DFTOutputFormat,
        coeffs_meta: CoeffsMeta,
    ) -> anyhow::Result<Self> {
        let plan = Self {
            kind,
            schedule: schedule.into(),
            format,
            scaling: None,
            bit_reversed: false,
            coeffs_meta,
        };
        plan.schedule.check("DFTPlan")?;
        if plan.num_factors().checked_mul(plan.coeffs_meta.log_delta()).is_none() {
            return Err(crate::CKKSCompositionError::InvalidPlan {
                plan: "DFTPlan",
                reason: format!(
                    "{} factors at log_delta {} overflow the consumed-bit count",
                    plan.num_factors(),
                    plan.coeffs_meta.log_delta(),
                ),
            }
            .into());
        }
        Ok(plan)
    }

    /// Sets the constant the matrix is multiplied by (default `1.0`).
    ///
    /// Errors if `scaling` is non-finite or `<= 0` — such a value would
    /// silently NaN-poison every generated diagonal via the per-factor
    /// `scaling^(1/depth)` spread.
    pub fn with_scaling(mut self, scaling: f64) -> anyhow::Result<Self> {
        if !scaling.is_finite() || scaling <= 0.0 {
            return Err(crate::CKKSCompositionError::InvalidPlan {
                plan: "DFTPlan",
                reason: format!("scaling must be finite and > 0, got {scaling}"),
            }
            .into());
        }
        self.scaling = Some(scaling);
        Ok(self)
    }

    /// Applies the transform bit-reversed (and expects bit-reversed inputs).
    /// Default `false`.
    pub fn with_bit_reversed(mut self, bit_reversed: bool) -> Self {
        self.bit_reversed = bit_reversed;
        self
    }

    /// Encode (IDFT) or Decode (DFT).
    pub fn kind(&self) -> DFTType {
        self.kind
    }

    /// The factorization schedule, in evaluation order.
    pub fn schedule(&self) -> &FactorSchedule {
        &self.schedule
    }

    /// Post-processing format (canonical on a resolved plan — see the field
    /// docs on the struct).
    pub fn format(&self) -> DFTOutputFormat {
        self.format
    }

    /// Constant the matrix is multiplied by; `None` = `1.0`.
    pub fn scaling(&self) -> Option<f64> {
        self.scaling
    }

    /// Whether the transform is applied bit-reversed.
    pub fn bit_reversed(&self) -> bool {
        self.bit_reversed
    }

    /// Per-factor coefficient metadata (its `log_delta` is the per-factor
    /// scale).
    pub fn coeffs_meta(&self) -> CoeffsMeta {
        self.coeffs_meta
    }

    /// `log2` of the number of complex slots the transform acts on: the sum of
    /// the factorization schedule (total FFT layers).
    pub fn log_slots(&self) -> usize {
        self.schedule.log_slots()
    }

    /// Number of factor matrices (the schedule length).
    pub fn num_factors(&self) -> usize {
        self.schedule.num_factors()
    }

    pub fn consumed_bits(&self) -> usize {
        self.num_factors() * self.coeffs_meta.log_delta()
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
    ///
    /// Infallible: every `DFTPlan` is shape-valid by construction
    /// ([`Self::new`]).
    pub fn diagonal_indexes(&self, log_n: usize) -> Vec<Vec<i64>> {
        let log_slots = self.log_slots();
        let slots = 1i64 << log_slots;
        let sparse = self.is_sparse_repack(log_n);

        let mut out = Vec::with_capacity(self.num_factors());
        let mut fft_level = log_slots;
        for (i, m) in self.schedule.steps.iter().map(|s| s.depth).enumerate() {
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
                let rot = dft_layer_rotation(self.kind, self.bit_reversed, level, log_slots, mask);
                let mut next: BTreeSet<i64> = BTreeSet::new();
                for &d in &set {
                    next.extend(dft_layer_spread(d, rot, mask));
                }
                set = next;
                level -= 1;
            }
            out.push(set.into_iter().collect());
            fft_level -= m;
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
    ///
    /// Infallible: every `DFTPlan` is shape-valid by construction
    /// ([`Self::new`]).
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let sparse = self.is_sparse_repack(log_n);
        let slots = 1i64 << self.log_slots();
        // The factor diagonals (and thus the BSGS schedule) live in the `2·slots`
        // working ring on the sparse path, else in `slots`.
        let slot_count = if sparse { (slots << 1) as usize } else { slots as usize };

        let mut set: BTreeSet<i64> = BTreeSet::new();
        for (indexes, step) in self.diagonal_indexes(log_n).into_iter().zip(&self.schedule.steps) {
            let layout = LinearTransformationLayout {
                indexes,
                slots: slot_count,
                strategy: LinearTransformationStrategy::Bsgs {
                    giant_step: step.giant_step,
                },
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
        self.inner.plan.coeffs_meta.meta
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
