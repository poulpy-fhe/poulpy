use std::{fmt, result};

use crate::security::SecretSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoreError {
    PendingLinearTermsOverflow {
        op: &'static str,
        base2k: usize,
        pending_linear_terms: usize,
        max_pending_linear_terms: usize,
    },
    NonNormalizedCiphertextInput {
        op: &'static str,
        base2k: usize,
        pending_linear_terms: usize,
    },
    /// A GGLWE key-use policy, resolver, or registry rejected a request.
    GGLWEKeyUse { op: &'static str, detail: String },
    /// A [`ReplicatedLayout`](crate::layouts::ReplicatedLayout) violates its invariants.
    InvalidReplicatedLayout { detail: &'static str },
    /// A [`SecurityTable`](crate::security::SecurityTable) has no bound for a
    /// secret at `target_bits` bits of security: the table lacks the target, the
    /// secret's distribution, its degree, or a result for its cell. The message names
    /// neither the secret's distribution nor its degree, which stay in `secret`,
    /// since key generation reads them from a secret operand.
    SecurityUnsupported {
        target_bits: usize,
        secret: SecretSpec,
        reason: &'static str,
    },
    /// The modulus of an object, named by `what`, is above the bound of its secret
    /// at `target_bits` bits of security. The message names neither the secret nor its
    /// bound, which depends on it: both stay in the fields.
    SecurityBoundExceeded {
        target_bits: usize,
        secret: SecretSpec,
        what: String,
        log_q: usize,
        max_log_q: usize,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::PendingLinearTermsOverflow {
                op,
                base2k,
                pending_linear_terms,
                max_pending_linear_terms,
            } => write!(
                f,
                "{op}: pending_linear_terms={pending_linear_terms} exceeds safe bound {max_pending_linear_terms} for base2k={base2k}"
            ),
            CoreError::NonNormalizedCiphertextInput {
                op,
                base2k,
                pending_linear_terms,
            } => write!(
                f,
                "{op}: ciphertext is not normalized (base2k={base2k}, pending_linear_terms={pending_linear_terms})"
            ),
            CoreError::GGLWEKeyUse { op, detail } => write!(f, "{op}: {detail}"),
            CoreError::InvalidReplicatedLayout { detail } => write!(f, "invalid replicated layout: {detail}"),
            CoreError::SecurityUnsupported { target_bits, reason, .. } => {
                write!(f, "{target_bits}-bit security cannot be certified for this secret: {reason}")
            }
            CoreError::SecurityBoundExceeded {
                target_bits,
                what,
                log_q,
                ..
            } => write!(
                f,
                "{what}: a modulus of {log_q} bits exceeds the {target_bits}-bit bound for its secret"
            ),
        }
    }
}

impl std::error::Error for CoreError {}

pub type Result<T> = result::Result<T, CoreError>;
