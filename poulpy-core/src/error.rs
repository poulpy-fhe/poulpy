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
        }
    }
}

impl std::error::Error for CoreError {}

pub type Result<T> = result::Result<T, CoreError>;
