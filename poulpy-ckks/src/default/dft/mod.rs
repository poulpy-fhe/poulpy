//! Homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) — default implementation.
//!
//! Phase 1 hosts the scheme-free factor-matrix generator ([`matrices`]); the
//! evaluation layer (encode factors, chain linear transforms + rescale, the
//! real/imag format wrappers) lands in later increments. See
//! `docs/ckks_dft.md`.

pub mod eval;
pub mod matrices;

pub use eval::{
    DftFactor, ckks_coeffs_to_slots_assign, ckks_coeffs_to_slots_repack, ckks_coeffs_to_slots_split, ckks_dft_evaluate_assign,
    ckks_new_dft_matrix_prepared, ckks_new_dft_matrix, ckks_slots_to_coeffs_assign, ckks_slots_to_coeffs_repack,
    ckks_slots_to_coeffs_split,
};
pub use matrices::gen_dft_matrices;
