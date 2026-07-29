//! Backend-independent CKKS encoding algorithms.

pub(crate) mod paco;
pub(crate) mod ship;
pub use paco::coeff_enc::paco_coeff_encodings_host;
pub use ship::coeff_enc::ship_coeff_encodings_host;
