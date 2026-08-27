use std::{error::Error, fmt};

/// Typed error returned by every fallible CKKS **operation** (the `api` op
/// traits and the layers backing them).
///
/// Callers branch on [`CKKSError::Composition`] for recoverable semantic
/// conditions — e.g. [`CKKSCompositionError::InsufficientHomomorphicCapacity`]
/// means "bootstrap now"; the layout/plan mismatch variants mean caller misuse.
/// [`CKKSError::Internal`] carries an [`anyhow::Error`] context chain for
/// invariant or backend failures and is not intended for matching.
///
/// Setup-time constructors (plan compilation, key-set assembly) still return
/// `anyhow::Result`; typed variants they raise internally are recovered by the
/// downcasting [`From<anyhow::Error>`] bridge when they cross an op boundary.
#[derive(Debug)]
#[non_exhaustive]
pub enum CKKSError {
    /// Semantic composition/alignment failure; match on the inner
    /// [`CKKSCompositionError`] to branch on the condition.
    Composition(CKKSCompositionError),
    /// Internal invariant or backend failure, with its context chain.
    Internal(anyhow::Error),
}

impl CKKSError {
    /// The semantic composition error, if this is one.
    pub fn composition(&self) -> Option<&CKKSCompositionError> {
        match self {
            Self::Composition(c) => Some(c),
            Self::Internal(_) => None,
        }
    }
}

impl fmt::Display for CKKSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Composition(c) => fmt::Display::fmt(c, f),
            // `{:#}` renders anyhow's full context chain on one line.
            Self::Internal(e) => write!(f, "{e:#}"),
        }
    }
}

impl Error for CKKSError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Composition(_) => None,
            Self::Internal(e) => Some(e.as_ref()),
        }
    }
}

impl From<CKKSCompositionError> for CKKSError {
    fn from(e: CKKSCompositionError) -> Self {
        Self::Composition(e)
    }
}

/// Bridge from `anyhow`-based internals: a [`CKKSCompositionError`] anywhere in
/// the chain is recovered as [`CKKSError::Composition`] (so typed conditions
/// raised inside setup/compile helpers stay matchable); anything else becomes
/// [`CKKSError::Internal`] with its context preserved.
impl From<anyhow::Error> for CKKSError {
    fn from(e: anyhow::Error) -> Self {
        match e.downcast::<CKKSCompositionError>() {
            Ok(c) => Self::Composition(c),
            Err(e) => Self::Internal(e),
        }
    }
}

/// Result type of every fallible CKKS operation.
pub type CKKSResult<T> = core::result::Result<T, CKKSError>;

/// `bail!` for [`CKKSResult`] contexts: formats into [`CKKSError::Internal`].
macro_rules! ckks_bail {
    ($($arg:tt)*) => {
        return Err($crate::CKKSError::Internal(::anyhow::anyhow!($($arg)*)))
    };
}

/// `ensure!` for [`CKKSResult`] contexts: formats into [`CKKSError::Internal`].
macro_rules! ckks_ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::CKKSError::Internal(::anyhow::anyhow!($($arg)*)));
        }
    };
}

pub(crate) use {ckks_bail, ckks_ensure};

type Result<T> = CKKSResult<T>;

/// CKKS composition and alignment errors returned by high-level operations.
///
/// These errors describe semantic failures such as insufficient precision,
/// incompatible plaintext/ciphertext layouts, or metadata that cannot fit in
/// the requested output storage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CKKSCompositionError {
    /// Shrinking a ciphertext buffer would drop required semantic bits.
    LimbReallocationShrinksBelowMetadata {
        max_k: usize,
        log_delta: usize,
        base2k: usize,
        requested_limbs: usize,
    },
    /// An operation requires more `log_budget` than is still available.
    InsufficientHomomorphicCapacity {
        op: &'static str,
        available_log_budget: usize,
        required_bits: usize,
    },
    /// A plaintext and ciphertext use different limb radices.
    PlaintextBase2KMismatch {
        op: &'static str,
        ct_base2k: usize,
        pt_base2k: usize,
    },
    /// A full plaintext-vector operation received a plaintext with the wrong degree.
    PlaintextDegreeMismatch { op: &'static str, ct_n: usize, pt_n: usize },
    /// A plaintext-coefficient operation requested a coefficient outside the source or destination layout.
    PlaintextCoefficientOutOfRange {
        op: &'static str,
        role: &'static str,
        coeff: usize,
        n: usize,
    },
    /// A requested rotation/conjugation key is not present in the provided key map.
    MissingAutomorphismKey { op: &'static str, rotation: i64, k: u32 },
    /// No relinearization key covers the exact precision the operation works at.
    MissingRelinearizationKey { op: &'static str, k: u32 },
    /// A plaintext cannot be aligned into the requested destination precision.
    PlaintextAlignmentImpossible {
        op: &'static str,
        ct_log_budget: usize,
        pt_log_delta: usize,
        pt_k: usize,
    },
    /// A multiplication would consume more semantic precision than available.
    MultiplicationPrecisionUnderflow {
        op: &'static str,
        lhs_log_budget: usize,
        rhs_log_budget: usize,
        lhs_log_delta: usize,
        rhs_log_delta: usize,
    },
    /// A plan literal (DFT, EvalMod, bootstrapping) violates a shape or
    /// numeric invariant required to compile or evaluate it.
    InvalidPlan { plan: &'static str, reason: String },
    /// A prepared right multiplication operand was built under a different
    /// ring degree, limb radix, or rank than the destination it is applied to.
    PreparedOperandLayoutMismatch {
        op: &'static str,
        dst_n: usize,
        dst_base2k: usize,
        dst_rank: usize,
        prep_n: usize,
        prep_base2k: usize,
        prep_rank: usize,
    },
    /// A homomorphic-DFT evaluation received a prepared matrix whose
    /// kind/format/sparsity does not match the entry point.
    DftMatrixMismatch {
        op: &'static str,
        expected: &'static str,
        got: String,
    },
}

impl fmt::Display for CKKSCompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimbReallocationShrinksBelowMetadata {
                max_k,
                log_delta,
                base2k,
                requested_limbs,
            } => write!(
                f,
                "cannot reallocate to {requested_limbs} limbs: requested capacity is {} bits but ciphertext metadata requires a larger buffer (max_k={max_k}, log_delta={log_delta}, base2k={base2k})",
                requested_limbs * base2k,
            ),
            Self::InsufficientHomomorphicCapacity {
                op,
                available_log_budget,
                required_bits,
            } => write!(
                f,
                "{op} cannot consume {required_bits} bits of log_budget: only {available_log_budget} bits remain"
            ),
            Self::PlaintextBase2KMismatch {
                op,
                ct_base2k,
                pt_base2k,
            } => write!(
                f,
                "{op} requires matching base2k values, got ciphertext base2k={ct_base2k} and plaintext base2k={pt_base2k}"
            ),
            Self::PlaintextDegreeMismatch { op, ct_n, pt_n } => {
                write!(
                    f,
                    "{op} requires a full plaintext with degree {ct_n}, got plaintext degree {pt_n}"
                )
            }
            Self::PlaintextCoefficientOutOfRange { op, role, coeff, n } => {
                write!(f, "{op} coefficient index {coeff} is out of range for {role} degree {n}")
            }
            Self::MissingAutomorphismKey { op, rotation, k } => {
                write!(
                    f,
                    "{op} requires an automorphism key for rotation {rotation} at precision k={k}, but none was provided"
                )
            }
            Self::MissingRelinearizationKey { op, k } => {
                write!(
                    f,
                    "{op} requires a relinearization key at precision k={k}, but none was provided"
                )
            }
            Self::PlaintextAlignmentImpossible {
                op,
                ct_log_budget,
                pt_log_delta,
                pt_k,
            } => write!(
                f,
                "{op} cannot align plaintext with ciphertext: ct.log_budget + pt.log_delta = {} but plaintext precision is {pt_k} bits (ct.log_budget={ct_log_budget}, pt.log_delta={pt_log_delta})",
                ct_log_budget + pt_log_delta
            ),
            Self::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_budget,
                rhs_log_budget,
                lhs_log_delta,
                rhs_log_delta,
            } => write!(
                f,
                "{op} cannot compose inputs: min(log_budget)={} is smaller than min(log_delta)={} (lhs: log_budget={lhs_log_budget}, log_delta={lhs_log_delta}; rhs: log_budget={rhs_log_budget}, log_delta={rhs_log_delta})",
                lhs_log_budget.min(rhs_log_budget),
                lhs_log_delta.min(rhs_log_delta)
            ),
            Self::InvalidPlan { plan, reason } => {
                write!(f, "invalid {plan}: {reason}")
            }
            Self::PreparedOperandLayoutMismatch {
                op,
                dst_n,
                dst_base2k,
                dst_rank,
                prep_n,
                prep_base2k,
                prep_rank,
            } => write!(
                f,
                "{op} received a prepared operand from a different layout: destination (n={dst_n}, base2k={dst_base2k}, rank={dst_rank}) vs prepared (n={prep_n}, base2k={prep_base2k}, rank={prep_rank})"
            ),
            Self::DftMatrixMismatch { op, expected, got } => {
                write!(f, "{op} requires a {expected} DFT matrix, got {got}")
            }
        }
    }
}

impl Error for CKKSCompositionError {}

pub(crate) fn checked_log_budget_sub(op: &'static str, available_log_budget: usize, required_bits: usize) -> Result<usize> {
    available_log_budget.checked_sub(required_bits).ok_or_else(|| {
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op,
            available_log_budget,
            required_bits,
        }
        .into()
    })
}

pub(crate) fn ensure_base2k_match(op: &'static str, ct_base2k: usize, pt_base2k: usize) -> Result<()> {
    if ct_base2k != pt_base2k {
        return Err(CKKSCompositionError::PlaintextBase2KMismatch {
            op,
            ct_base2k,
            pt_base2k,
        }
        .into());
    }
    Ok(())
}

pub(crate) fn ensure_plaintext_degree_match(op: &'static str, ct_n: usize, pt_n: usize) -> Result<()> {
    if ct_n != pt_n {
        return Err(CKKSCompositionError::PlaintextDegreeMismatch { op, ct_n, pt_n }.into());
    }
    Ok(())
}

pub(crate) fn ensure_plaintext_coeff_in_range(op: &'static str, role: &'static str, coeff: usize, n: usize) -> Result<()> {
    if coeff >= n {
        return Err(CKKSCompositionError::PlaintextCoefficientOutOfRange { op, role, coeff, n }.into());
    }
    Ok(())
}

pub(crate) fn ensure_plaintext_alignment(op: &'static str, ct_log_budget: usize, pt_log_delta: usize, pt_k: usize) -> Result<()> {
    let available = ct_log_budget + pt_log_delta;
    if available < pt_k {
        return Err(CKKSCompositionError::PlaintextAlignmentImpossible {
            op,
            ct_log_budget,
            pt_log_delta,
            pt_k,
        }
        .into());
    }
    Ok(())
}

pub(crate) fn checked_mul_ct_log_budget(
    op: &'static str,
    lhs_log_budget: usize,
    rhs_log_budget: usize,
    lhs_log_delta: usize,
    rhs_log_delta: usize,
) -> Result<usize> {
    // Bivariate-torus CKKS multiplication returns the already-rescaled product:
    // the output scale is `min(log_delta)` and the discarded scale/headroom is
    // therefore `max(log_delta)`, not `lhs_log_delta + rhs_log_delta` as it
    // would be for an unreduced fixed-point product. See the `CKKSMulOps`
    // metadata docs and the bivariate Torus analysis cited in the README
    // ("Revisiting Key Decomposition Techniques for FHE", ePrint 2023/771).
    lhs_log_budget
        .min(rhs_log_budget)
        .checked_sub(lhs_log_delta.max(rhs_log_delta))
        .ok_or_else(|| {
            CKKSCompositionError::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_budget,
                rhs_log_budget,
                lhs_log_delta,
                rhs_log_delta,
            }
            .into()
        })
}

pub(crate) fn checked_mul_pt_log_budget(
    op: &'static str,
    lhs_log_budget: usize,
    rhs_log_budget: usize,
    lhs_log_delta: usize,
    rhs_log_delta: usize,
) -> Result<usize> {
    lhs_log_budget.checked_sub(rhs_log_delta).ok_or_else(|| {
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op,
            lhs_log_budget,
            rhs_log_budget,
            lhs_log_delta,
            rhs_log_delta,
        }
        .into()
    })
}
