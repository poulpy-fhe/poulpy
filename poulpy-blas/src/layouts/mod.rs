//! Grids of block ciphertexts, the encoded per-block maps and their host diagonals, the prepared operands of products,
//! and the checks every operation runs first.

pub mod grid;
pub mod maps;

pub(crate) mod ciphertext;
pub(crate) mod context;
pub(crate) mod encoding;
pub(crate) mod matmul_operand;
pub(crate) mod pt_operand;
pub(crate) mod validate;

pub use ciphertext::{
    BlasBytesOf, BlasCiphertext, BlasCiphertextOwned, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasMatMulOperandInfos,
    BlasMatMulOperandLayout, BlasModuleAlloc, grid_batch, replicated,
};
pub use context::{
    BlasContext, BlasContextInfos, BlasContextLayout, BlasContextPreparedFactory, BlasContextUnprepared, BlasPlan,
    BlasPtAlgorithm, sum_galois_elements,
};
pub use encoding::BlasEncodingOps;
pub use matmul_operand::{
    BlasMatMulLeft, BlasMatMulOperand, BlasMatMulOperandPreparedFactory, BlasMatMulRight, blas_matmul_left_operand_layout,
    blas_matmul_left_term_layout, blas_matmul_right_operand_layout, blas_matmul_right_term_layout,
};
pub use pt_operand::{
    BlasPtOperand, BlasPtOperandInfos, BlasPtOperandLayout, BlasPtOperandPreparedFactory, BlasPtOperandUnprepared, BlasPtSide,
    Left, Right,
};
pub use validate::{
    validate_blas_matmul, validate_blas_matmul_prepare_left, validate_blas_matmul_prepare_right, validate_blas_matmul_prepared,
    validate_blas_matmul_pt_left, validate_blas_matmul_pt_right, validate_blas_rotate, validate_blas_rotate_cols,
    validate_blas_rotate_rows, validate_blas_sum, validate_blas_transpose,
};
