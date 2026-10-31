//! Backend-independent CKKS encoding algorithms.

mod linear_transformation;
pub(crate) mod paco;
pub(crate) mod ship;
pub use linear_transformation::{
    ckks_linear_transformation_encode_from_diagonals, ckks_linear_transformation_encode_into,
    ckks_linear_transformation_encode_tmp_bytes,
};
pub use paco::coeff_enc::ckks_paco_coeff_encodings_host;
pub use ship::coeff_enc::ckks_ship_coeff_encodings_host;
