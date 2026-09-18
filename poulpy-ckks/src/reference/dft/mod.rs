//! Homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) — default implementation.
//!
//! Phase 1 hosts the scheme-free factor-matrix generator ([`matrices`]); the
//! evaluation layer (encode factors, chain linear transforms + rescale, the
//! real/imag format wrappers) lands in later increments. The homomorphic DFT is
//! documented as a stage of the bootstrapping pipeline in
//! [`docs/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/bootstrapping.md).

pub mod eval;
pub mod matrices;

pub use eval::{
    ckks_coeffs_to_slots_assign, ckks_coeffs_to_slots_repack, ckks_coeffs_to_slots_split, ckks_dft_evaluate_assign,
    ckks_new_dft_matrix, ckks_prepare_dft_matrix, ckks_slots_to_coeffs_assign, ckks_slots_to_coeffs_repack,
    ckks_slots_to_coeffs_split,
};
pub use matrices::{DftScalar, gen_dft_matrices, gen_dft_matrices_blockwise};
