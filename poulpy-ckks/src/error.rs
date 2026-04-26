use std::{error::Error, fmt};

use anyhow::Result;

/// CKKS composition and alignment errors returned by high-level operations.
///
/// These errors describe semantic failures such as insufficient precision,
/// incompatible plaintext/ciphertext layouts, or metadata that cannot fit in
/// the requested output storage.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// A requested rotation/conjugation key is not present in the provided key map.
    MissingAutomorphismKey { op: &'static str, rotation: i64 },
    /// A plaintext cannot be aligned into the requested destination precision.
    PlaintextAlignmentImpossible {
        op: &'static str,
        ct_log_budget: usize,
        pt_log_delta: usize,
        pt_max_k: usize,
    },
    /// A multiplication would consume more semantic precision than available.
    MultiplicationPrecisionUnderflow {
        op: &'static str,
        lhs_log_budget: usize,
        rhs_log_budget: usize,
        lhs_log_delta: usize,
        rhs_log_delta: usize,
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
            Self::MissingAutomorphismKey { op, rotation } => {
                write!(
                    f,
                    "{op} requires an automorphism key for rotation {rotation}, but none was provided"
                )
            }
            Self::PlaintextAlignmentImpossible {
                op,
                ct_log_budget,
                pt_log_delta,
                pt_max_k,
            } => write!(
                f,
                "{op} cannot align plaintext with ciphertext: ct.log_budget + pt.log_delta = {} but pt.max_k = {pt_max_k} (ct.log_budget={ct_log_budget}, pt.log_delta={pt_log_delta})",
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

pub(crate) fn ensure_plaintext_alignment(
    op: &'static str,
    ct_log_budget: usize,
    pt_log_delta: usize,
    pt_max_k: usize,
) -> Result<usize> {
    let available = ct_log_budget + pt_log_delta;
    if available < pt_max_k {
        return Err(CKKSCompositionError::PlaintextAlignmentImpossible {
            op,
            ct_log_budget,
            pt_log_delta,
            pt_max_k,
        }
        .into());
    }
    Ok(available - pt_max_k)
}

pub(crate) fn checked_mul_ct_log_budget(
    op: &'static str,
    lhs_log_budget: usize,
    rhs_log_budget: usize,
    lhs_log_delta: usize,
    rhs_log_delta: usize,
) -> Result<usize> {
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
