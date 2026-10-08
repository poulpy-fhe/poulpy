//! Registration of the HAL traits. Kernels live in their own modules; these
//! files only turn layouts into kernel operands and pick the launches.

mod convolution;
mod operand;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;

// Only the tests reach the two core products until the core hooks are registered.
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use convolution::glwe_mul_columns;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use vmp::gglwe_product_digits_strided;
