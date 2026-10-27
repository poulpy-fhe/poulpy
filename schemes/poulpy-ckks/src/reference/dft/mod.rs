//! Homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) — reference implementation.
//!
//! Re-exports the scheme-free factor-matrix generator ([`gen_dft_matrices`]) and
//! holds the evaluation layer: factor encoding, the chain of linear transformations
//! and the real/imag format wrappers. The homomorphic DFT is documented as a stage
//! of the bootstrapping pipeline in
//! [`docs/guides/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/guides/bootstrapping.md).

pub mod eval;

pub use crate::layouts::dft_matrices::{gen_dft_matrices, gen_dft_matrices_blockwise};
pub use eval::{
    ckks_dft_evaluate_assign, ckks_dft_tmp_bytes, ckks_new_dft_matrix, ckks_new_dft_matrix_tmp_bytes, ckks_prepare_dft_matrix,
    ckks_prepare_dft_matrix_tmp_bytes,
};
