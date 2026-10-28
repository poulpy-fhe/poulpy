//! Encrypted matrix arithmetic on poulpy-ckks.
//!
//! A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block. A ciphertext with `S` slots
//! holds `S / d^2` blocks, matrix slot `s` holding entry `(r, c)` of its block at slot `s d^2 + r d + c`, so every
//! operation acts on a batch of matrices: up to `S / d^2`, repeated over the slots, which the grid records
//! ([`layouts::BlasCiphertext::from_blocks`]). Intra operations apply one map to every matrix of the batch; inter
//! operations move whole matrices across it. Products follow Jiang, Kim, Lauter and Song (2018), with every block map
//! encoded once at a caller-chosen precision.

pub mod api;
mod delegates;
pub mod layouts;
pub mod oep;
pub mod reference;
#[cfg(feature = "test-utils")]
#[expect(
    clippy::too_many_arguments,
    reason = "test fixtures pass their operands, keys and parameters one by one"
)]
pub mod test_suite;

pub use layouts::{
    BlasKeyUses, blas_matmul_key_uses, blas_matmul_max_base2k, blas_matmul_output_layout, blas_matmul_prepare_left_key_uses,
    blas_matmul_prepare_right_key_uses, blas_matmul_prepared_key_uses, blas_matmul_prepared_output_layout,
    blas_matmul_pt_left_key_uses, blas_matmul_pt_left_max_base2k, blas_matmul_pt_left_output_layout,
    blas_matmul_pt_right_key_uses, blas_matmul_pt_right_max_base2k, blas_matmul_pt_right_output_layout,
    blas_rotate_cols_key_uses, blas_rotate_cols_output_layout, blas_rotate_key_uses, blas_rotate_output_layout,
    blas_rotate_rows_key_uses, blas_rotate_rows_output_layout, blas_sum_key_uses, blas_sum_output_layout,
    blas_transpose_key_uses, blas_transpose_output_layout,
};
