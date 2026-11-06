//! Grids of block ciphertexts, the encoded per-block maps and their host diagonals, the prepared operands of products,
//! the checks every operation runs first, and the planning queries: the keys an operation looks up, the layout it
//! stamps on `res`, and the largest radix of a product.

pub mod grid;
pub mod maps;

pub(crate) mod ciphertext;
pub(crate) mod context;
pub(crate) mod diagonal_ciphertext;
pub(crate) mod encoding;
pub(crate) mod line;
pub(crate) mod matmul_operand;
pub(crate) mod matvec_operand;
pub(crate) mod planning;
pub(crate) mod pt_operand;
pub(crate) mod validate;
pub(crate) mod vector;

pub use ciphertext::{
    BlasCiphertext, BlasCiphertextAlloc, BlasCiphertextBytesOf, BlasCiphertextOwned, BlasGrid, BlasGridInfos, BlasGridMut,
    BlasLayout, BlasMatMulOperandInfos, BlasMatMulOperandLayout, grid_batch, replicated,
};
pub use context::{
    BlasContext, BlasContextAlloc, BlasContextInfos, BlasContextLayout, BlasContextPrepared, BlasPlan, BlasPtAlgorithm,
    sum_galois_elements,
};
pub use diagonal_ciphertext::{
    BlasDiagonalCiphertext, BlasDiagonalCiphertextAlloc, BlasDiagonalCiphertextBytesOf, BlasDiagonalCiphertextInfos,
    BlasDiagonalCiphertextLayout, BlasDiagonalCiphertextMap, BlasDiagonalCiphertextPrepared, append_col_diagonal_galois_elements,
    append_row_diagonal_galois_elements, matvec_diagonal_galois_elements,
};
pub use line::BlasGridLine;
pub use matmul_operand::{
    BlasMatMulLeftOperandPrepared, BlasMatMulOperandPrepared, BlasMatMulRightOperandPrepared, blas_matmul_left_operand_layout,
    blas_matmul_left_term_layout, blas_matmul_right_operand_layout, blas_matmul_right_term_layout,
};
pub use matvec_operand::{
    BlasMatVecMaps, BlasMatVecOperand, BlasMatVecOperandAlloc, BlasMatVecOperandInfos, BlasMatVecOperandLayout,
    BlasMatVecOperandPrepared, matvec_pt_galois_elements,
};
pub use planning::{
    BlasKeyUses, blas_append_col_diagonal_key_uses, blas_append_row_diagonal_key_uses, blas_grid_to_vector_key_uses,
    blas_grid_to_vector_max_base2k, blas_grid_to_vector_output_layout, blas_matmul_key_uses,
    blas_matmul_left_operand_prepare_key_uses, blas_matmul_max_base2k, blas_matmul_output_layout, blas_matmul_prepared_key_uses,
    blas_matmul_prepared_output_layout, blas_matmul_pt_left_key_uses, blas_matmul_pt_left_max_base2k,
    blas_matmul_pt_left_output_layout, blas_matmul_pt_right_key_uses, blas_matmul_pt_right_max_base2k,
    blas_matmul_pt_right_output_layout, blas_matmul_right_operand_prepare_key_uses, blas_matvec_diagonal_key_uses,
    blas_matvec_diagonal_max_base2k, blas_matvec_diagonal_output_layout, blas_matvec_pt_key_uses, blas_matvec_pt_max_base2k,
    blas_matvec_pt_output_layout, blas_rotate_cols_key_uses, blas_rotate_cols_output_layout, blas_rotate_key_uses,
    blas_rotate_output_layout, blas_rotate_rows_key_uses, blas_rotate_rows_output_layout, blas_sum_cols_key_uses,
    blas_sum_cols_output_layout, blas_sum_key_uses, blas_sum_output_layout, blas_sum_rows_key_uses, blas_sum_rows_output_layout,
    blas_transpose_key_uses, blas_transpose_output_layout, blas_vector_to_grid_key_uses, blas_vector_to_grid_max_base2k,
    blas_vector_to_grid_output_layout,
};
pub use pt_operand::{
    BlasPtOperand, BlasPtOperandAlloc, BlasPtOperandInfos, BlasPtOperandLayout, BlasPtOperandPrepared, BlasPtSide, Left, Right,
};
pub use validate::{
    validate_blas_append_col_diagonal, validate_blas_append_row_diagonal, validate_blas_grid_to_vector, validate_blas_matmul,
    validate_blas_matmul_left_operand_prepare, validate_blas_matmul_prepared, validate_blas_matmul_pt_left,
    validate_blas_matmul_pt_right, validate_blas_matmul_right_operand_prepare, validate_blas_matvec_diagonal,
    validate_blas_matvec_pt, validate_blas_rotate, validate_blas_rotate_cols, validate_blas_rotate_rows, validate_blas_sum,
    validate_blas_sum_cols, validate_blas_sum_rows, validate_blas_transpose, validate_blas_vector_to_grid,
};
pub use vector::{
    BlasVector, BlasVectorAlloc, BlasVectorBytesOf, BlasVectorInfos, BlasVectorLayout, BlasVectorMut, BlasVectorOwned,
    BlasVectorRef, BlasVectorShape, vector_period,
};
