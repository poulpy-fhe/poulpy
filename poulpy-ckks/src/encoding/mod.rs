//! Backend-independent CKKS encoding algorithms.

pub(crate) mod paco;
pub use paco::coeff_enc::paco_coeff_encodings_host;
