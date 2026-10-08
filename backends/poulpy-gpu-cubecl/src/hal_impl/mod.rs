//! Registration of the HAL traits. Kernels live in their own modules; these
//! files only turn layouts into kernel operands and pick the launches.

mod operand;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
