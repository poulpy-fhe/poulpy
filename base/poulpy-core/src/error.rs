use std::{fmt, result};

#[derive(Debug, Clone, PartialEq, Eq)]
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
        }
    }
}

impl std::error::Error for CoreError {}

pub type Result<T> = result::Result<T, CoreError>;
