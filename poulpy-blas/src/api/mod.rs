//! Public matrix operations on `Module<BE>`.

mod inter;
mod intra;
mod matmul;

pub use inter::BlasMatInterOps;
pub use intra::BlasMatIntraOps;
pub use matmul::BlasMatMulOps;
