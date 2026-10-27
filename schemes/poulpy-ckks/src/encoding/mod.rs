//! Backend-independent CKKS encoding algorithms.

pub(crate) mod linear_transformation;
pub(crate) mod paco;
pub(crate) mod ship;
pub use linear_transformation::{
    ckks_encode_linear_transformation_from_diagonals, ckks_encode_linear_transformation_into,
    ckks_encode_linear_transformation_tmp_bytes,
};
pub use paco::coeff_enc::paco_coeff_encodings_host;
pub use ship::coeff_enc::ship_coeff_encodings_host;
