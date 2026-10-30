//! Public matrix operations, and the encodings and preparations of contexts and public operands, on `Module<BE>`.

mod encoding;
mod inter;
mod intra;
mod matmul;
mod matvec;

pub use encoding::{
    BlasContextPreparedFactory, BlasDiagonalCiphertextPreparedFactory, BlasEncodingOps, BlasMatVecOperandPreparedFactory,
    BlasPtOperandPreparedFactory,
};
pub use inter::BlasMatInterOps;
pub use intra::BlasMatIntraOps;
pub use matmul::BlasMatMulOps;
pub use matvec::BlasMatVecOps;
