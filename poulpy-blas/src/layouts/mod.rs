//! Grids of block ciphertexts, the encoded per-block maps and their host diagonals.

pub mod grid;
pub mod maps;

mod ciphertext;
mod context;

pub use ciphertext::{
    BlasCiphertext, BlasCiphertextOwned, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasMatMulLeft, BlasMatMulRight,
    BlasModuleAlloc, grid_batch,
};
pub(crate) use ciphertext::{batch_fits, layout_of};
pub(crate) use context::sum_rotations;
pub use context::{BlasContext, BlasContextUnprepared, BlasPlan, BlasPtOperand, BlasPtOperandUnprepared, sum_galois_elements};
