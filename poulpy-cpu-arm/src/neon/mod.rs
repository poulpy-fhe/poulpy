//! Low-level NEON kernels for the `poulpy-cpu-arm` backend.

pub(crate) mod conv_i64;
pub(crate) mod fft;
pub(crate) mod normalize;
pub(crate) mod ntt120_arithmetic;
pub(crate) mod ntt120_convert;
pub(crate) mod ntt120_mat_vec;
pub(crate) mod ntt120_ntt;
pub(crate) mod q120;
pub(crate) mod reim4_arith;
pub(crate) mod reim4_conv;
pub(crate) mod reim_arith;
pub(crate) mod vec_znx_big;
pub(crate) mod znx;
pub(crate) mod znx_normalize;
