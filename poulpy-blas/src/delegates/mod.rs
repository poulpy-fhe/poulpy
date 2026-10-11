//! The API on `Module<BE>`: each call forwards to the backend's `*Impl`, then, on `Ok`, stamps the batch record of
//! `res` (a rejected call leaves it untouched). Products record the larger batch of their operands, the broadcast of
//! [`crate::api::BlasMatMulOps::blas_matmul`]; every other operation records its input's. Every call first runs its
//! `validate_*` from [`crate::reference`], which computes that batch, so implementations run only on inputs it
//! accepted. A preparation returns its operand, which records its input's batch itself.

mod inter;
mod intra;
mod matmul;
