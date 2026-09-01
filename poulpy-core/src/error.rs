use std::{fmt, result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// A GGLWE key-use policy, resolver, or registry rejected a request.
    GGLWEKeyUse { op: &'static str, detail: String },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::GGLWEKeyUse { op, detail } => write!(f, "{op}: {detail}"),
        }
    }
}

impl std::error::Error for CoreError {}

pub type Result<T> = result::Result<T, CoreError>;
