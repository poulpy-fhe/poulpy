//! Public matrix operations, and the encodings and preparations of contexts and public operands, on `Module<BE>`.

mod encoding;
mod inter;
mod intra;
mod matmul;

pub use encoding::{BlasContextPreparedFactory, BlasEncodingOps, BlasPtOperandPreparedFactory};
pub use inter::BlasMatInterOps;
pub use intra::BlasMatIntraOps;
pub use matmul::BlasMatMulOps;
