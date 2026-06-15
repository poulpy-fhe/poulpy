//! Homomorphic DFT parameters (CoeffsToSlots / SlotsToCoeffs).
//!
//! Scheme-level description of a factorized homomorphic (I)DFT. The factor
//! matrices themselves are generated from this literal by the `default::dft`
//! module; this file holds the (backend-free) parameter struct, its enums, and
//! the prepared [`DFTMatrix`] that carries the encoded/prepared factor operands.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the full
//! design.

use poulpy_core::{LinearTransformationPrepared, layouts::LinearTransformation};
use poulpy_hal::layouts::Backend;

use crate::{error::CKKSCompositionError, layouts::CKKSPlaintext};

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
    pub factor_giant_steps: Vec<usize>,
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
    pub factor_log_delta: usize,
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
        if self.factor_giant_steps.len() != self.factorization_depth.len() {
            return Err(format!(
                "invalid DFTPlan: factor_giant_steps (len {}) must match factorization_depth (len {})",
                self.factor_giant_steps.len(),
                self.factorization_depth.len()
            ));
        }
        if self.factor_giant_steps.contains(&0) {
            return Err(format!(
                "invalid DFTPlan: factor_giant_steps has a zero-width factor (use 1 for the direct schedule): {:?}",
                self.factor_giant_steps
            ));
        }
        Ok(())
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

/// A generated, ready-to-evaluate homomorphic (I)DFT, tagged by output format
/// and generic over the factor representation `R`.
///
/// The variant is the canonical witness of the matrix's output format, so the
/// three loosely-coupled `(kind, format, sparse)` flags collapse into one
/// exhaustively-matchable value and illegal combinations (e.g. sparse
/// `Standard`) are unrepresentable. The transform **direction**
/// ([`DFTType::Encode`]/[`Decode`]) stays in [`DFTPlan::kind`] and is checked at
/// the evaluation entry points; the format variant is what dispatch keys on.
///
/// `R` selects how each factor's RHS is stored, trading memory for compute:
/// the default `DFTMatrix<BE>` keeps unprepared plaintext diagonals
/// ([`LinearTransformation`]) and materializes them per factor at eval time
/// (minimal resident memory — for bandwidth-limited backends); the prepared
/// alias [`DFTMatrixPrepared`] keeps the convolution-domain diagonals resident
/// for faster repeated evaluation. The default builds via [`ckks_new_dft_matrix`]
/// and the prepared form via [`ckks_new_dft_matrix_prepared`]; both evaluate
/// through the same entry points (generic over `R`). The required Galois keys are
/// reported by `galois_elements`.
///
/// [`ckks_new_dft_matrix`]: crate::default::dft::ckks_new_dft_matrix
/// [`ckks_new_dft_matrix_prepared`]: crate::default::dft::ckks_new_dft_matrix_prepared
pub enum DFTMatrix<BE: Backend, R = LinearTransformation<CKKSPlaintext<<BE as Backend>::OwnedBuf>>> {
    /// Regular complex transform ([`DFTOutputFormat::Standard`]).
    Standard(DFTMatrixFactors<BE, R>),
    /// Real/imag split ([`DFTOutputFormat::SplitRealAndImag`], or a dense
    /// `RepackImagAsReal` canonicalized to it).
    Split(DFTMatrixFactors<BE, R>),
    /// Sparse `RepackImagAsReal` (imag repacked into the right half); the only
    /// variant carrying the `slots` repack rotation.
    Repack(DFTMatrixFactors<BE, R>),
}

/// Prepared (resident-RHS) [`DFTMatrix`]: each factor's diagonals are kept in the
/// convolution domain ([`LinearTransformationPrepared`]) rather than materialized
/// per factor, trading resident memory for faster repeated evaluation. Built via
/// [`ckks_new_dft_matrix_prepared`](crate::default::dft::ckks_new_dft_matrix_prepared).
pub type DFTMatrixPrepared<BE> = DFTMatrix<BE, LinearTransformationPrepared<BE>>;

/// Format discriminant an evaluation entry point requires, mirroring the
/// [`DFTMatrix`] variants. Paired with a [`DFTType`] direction in
/// [`DFTMatrix::ensure`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DftFormatTag {
    Standard,
    Split,
    Repack,
}

impl<BE: Backend, R> DFTMatrix<BE, R> {
    /// The factor operands common to every variant.
    pub(crate) fn inner(&self) -> &DFTMatrixFactors<BE, R> {
        match self {
            Self::Standard(f) | Self::Split(f) | Self::Repack(f) => f,
        }
    }

    /// The resolved plan (canonical `format`, populated `factor_log_delta`).
    pub fn plan(&self) -> &DFTPlan {
        &self.inner().plan
    }

    /// The per-factor right operands, in evaluation order.
    pub(crate) fn factor_operands(&self) -> &[R] {
        &self.inner().factors
    }

    /// Number of factor matrices (one prepared linear transformation each).
    pub fn num_factors(&self) -> usize {
        self.inner().factors.len()
    }

    /// `log_budget` bits consumed per factor (the per-factor plaintext `log_delta`).
    pub fn factor_log_delta(&self) -> usize {
        self.inner().plan.factor_log_delta
    }

    /// Total `log_budget` bits the whole transform consumes: `num_factors ×
    /// factor_log_delta`. The input ciphertext must have at least this much.
    pub fn consumed_bits(&self) -> usize {
        self.num_factors() * self.factor_log_delta()
    }

    /// Whether this is the sparse `RepackImagAsReal` path (needs the `slots`
    /// repack rotation and updates `log_sparsity`).
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Repack(_))
    }

    /// Validates that this matrix matches the `format`/`kind` an evaluation entry
    /// point requires, returning a descriptive error instead of panicking. The
    /// single gate every `coeffs_to_slots` / `slots_to_coeffs` variant goes
    /// through; the dense-`RepackImagAsReal`≡`Split` rule was already resolved at
    /// construction (it lives in the variant, not here).
    pub(crate) fn ensure(&self, op: &'static str, format: DftFormatTag, kind: DFTType) -> Result<(), CKKSCompositionError> {
        let format_ok = match format {
            DftFormatTag::Standard => matches!(self, Self::Standard(_)),
            DftFormatTag::Split => matches!(self, Self::Split(_)),
            DftFormatTag::Repack => matches!(self, Self::Repack(_)),
        };
        if format_ok && self.plan().kind == kind {
            return Ok(());
        }

        let expected: &'static str = match (kind, format) {
            (DFTType::Encode, DftFormatTag::Standard) => "CoeffsToSlots (Encode/Standard)",
            (DFTType::Encode, DftFormatTag::Split) => "CoeffsToSlotsSplit (Encode/Split)",
            (DFTType::Encode, DftFormatTag::Repack) => "CoeffsToSlotsRepack (Encode/sparse Repack)",
            (DFTType::Decode, DftFormatTag::Standard) => "SlotsToCoeffs (Decode/Standard)",
            (DFTType::Decode, DftFormatTag::Split) => "SlotsToCoeffsSplit (Decode/Split)",
            (DFTType::Decode, DftFormatTag::Repack) => "SlotsToCoeffsRepack (Decode/sparse Repack)",
        };
        Err(CKKSCompositionError::DftMatrixMismatch {
            op,
            expected,
            got: self.describe(),
        })
    }

    /// Human-readable `direction/format` description for error messages.
    fn describe(&self) -> String {
        let dir = match self.plan().kind {
            DFTType::Encode => "CoeffsToSlots",
            DFTType::Decode => "SlotsToCoeffs",
        };
        let fmt = match self {
            Self::Standard(_) => "Standard",
            Self::Split(_) => "Split",
            Self::Repack(_) => "sparse Repack",
        };
        format!("{dir}/{fmt}")
    }
}
