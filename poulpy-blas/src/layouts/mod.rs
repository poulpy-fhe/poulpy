//! Grids of block ciphertexts, the encoded per-block maps and their host diagonals.

pub mod grid;
pub mod maps;

mod ciphertext;
mod context;

pub(crate) use ciphertext::layout_of;
pub use ciphertext::{
    BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasLayout, BlasMatMulLeft, BlasMatMulRight, BlasModuleAlloc,
};
pub(crate) use context::sum_rotations;
pub use context::{BlasContext, BlasContextUnprepared, BlasPlan, BlasPtOperand, BlasPtOperandUnprepared, sum_galois_elements};
