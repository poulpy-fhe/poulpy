//! Homomorphic DFT parameters (CoeffsToSlots / SlotsToCoeffs).
//!
//! Scheme-level description of a factorized homomorphic (I)DFT. The factor
//! matrices themselves are generated from this literal by the `default::dft`
//! module; this file holds the (backend-free) parameter struct, its enums, and
//! the prepared [`DFTMatrix`] that carries the encoded/prepared factor operands.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the full
//! design.

use poulpy_hal::layouts::{Backend, galois_element};

use crate::api::PreparedLinearTransformationRhs;

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
pub enum DFTFormat {
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
/// `levels` is the factorization schedule. Its **sum** is the number of factor
/// matrices: the `log_slots` radix-2 FFT layers are distributed across `sum(levels)`
/// matrices, so each matrix merges roughly `log_slots / sum(levels)` layers. (The
/// partition itself — the individual `levels[i]` — only matters for RNS rescale
/// grouping, which has no analog in poulpy's bit-granular scale model; here a single
/// uniform per-factor scale is used, see `docs/ckks_dft.md` §6.) Must satisfy
/// `sum(levels) <= log_slots`.
#[derive(Clone, Debug)]
pub struct DFTMatrixLiteral {
    /// Encode (IDFT) or Decode (DFT).
    pub kind: DFTType,
    /// `log2` of the number of complex slots the transform acts on.
    pub log_slots: usize,
    /// Factorization schedule; `sum()` = number of factor matrices. Must satisfy
    /// `sum(levels) <= log_slots`. See the struct docs.
    pub levels: Vec<usize>,
    /// Post-processing format. Default [`DFTFormat::Standard`].
    pub format: DFTFormat,
    /// Constant the matrix is multiplied by. Default `1.0`.
    pub scaling: Option<f64>,
    /// If true, applies the transform bit-reversed (and expects bit-reversed
    /// inputs). Default false.
    pub bit_reversed: bool,
}

impl DFTMatrixLiteral {
    /// Number of factor matrices (`sum(levels)`) when `actual == false`, or the
    /// schedule length (`levels.len()`) when `actual == true`.
    pub fn depth(&self, actual: bool) -> usize {
        if actual { self.levels.len() } else { self.levels.iter().sum() }
    }

    /// Validates the basic shape invariant shared by generation and evaluation.
    pub fn check(&self) -> Result<(), String> {
        let max_depth = self.depth(false);
        if self.log_slots < max_depth {
            return Err(format!(
                "invalid DFTMatrixLiteral: log_slots={} < factorization depth={}",
                self.log_slots, max_depth
            ));
        }
        Ok(())
    }
}

/// A generated, ready-to-evaluate homomorphic (I)DFT.
///
/// Holds the prepared right operands (convolution-domain diagonals) of each
/// factor matrix, in evaluation order, plus the per-factor scale. Built once via
/// `ckks_new_dft_matrix`; the evaluator chains one prepared linear transformation
/// per factor (no explicit rescale — the torus plaintext-multiply realigns to the
/// input `log_delta`). The required Galois keys are reported by
/// [`Self::galois_elements`].
pub struct DFTMatrix<BE: Backend> {
    /// The parameters this matrix was generated from.
    pub literal: DFTMatrixLiteral,
    /// Prepared right operands, one per factor matrix, in evaluation order.
    pub(crate) factors: Vec<PreparedLinearTransformationRhs<BE>>,
    /// The per-factor plaintext scale: each factor consumes this many bits of
    /// `log_budget`.
    pub(crate) factor_log_delta: usize,
    /// True for the sparse `RepackImagAsReal` path (`log_slots < log_max_slots`):
    /// the imag-into-right-half repack needs an extra rotation by `slots` and the
    /// output's `log_sparsity` drops by one.
    pub(crate) sparse: bool,
}

impl<BE: Backend> DFTMatrix<BE> {
    /// Number of factor matrices (one prepared linear transformation each).
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// `log_budget` bits consumed per factor (the per-factor plaintext `log_delta`).
    pub fn factor_log_delta(&self) -> usize {
        self.factor_log_delta
    }

    /// Total `log_budget` bits the whole transform consumes: `num_factors ×
    /// factor_log_delta`. The input ciphertext must have at least this much.
    pub fn consumed_bits(&self) -> usize {
        self.factors.len() * self.factor_log_delta
    }

    /// Whether this is the sparse `RepackImagAsReal` path (needs the `slots`
    /// repack rotation and updates `log_sparsity`).
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// The distinct Galois elements whose automorphism keys evaluating this
    /// transform requires (the union over all factors, plus the `slots` repack
    /// rotation for the sparse path).
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for f in &self.factors {
            // `galois_elements` already maps non-zero baby/giant rotations to
            // Galois elements; union across factors.
            set.extend(f.galois_elements(cyclotomic_order));
        }
        if self.sparse {
            set.insert(galois_element(1i64 << self.literal.log_slots, cyclotomic_order));
        }
        // Defensive: drop the identity element if present.
        set.remove(&galois_element(0, cyclotomic_order));
        set.into_iter().collect()
    }
}
