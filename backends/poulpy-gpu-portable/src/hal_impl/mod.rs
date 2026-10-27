//! Registration of the HAL traits. Kernels live in their own modules; these
//! files only turn layouts into kernel operands and pick the launches.

mod convolution;
mod operand;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;

// Without the core registration only the device tests reach the two core products.
#[cfg(any(test, feature = "enable-core"))]
#[cfg_attr(not(feature = "enable-core"), allow(unused_imports))]
pub(crate) use convolution::glwe_mul_columns;
#[cfg(feature = "enable-core")]
pub(crate) use operand::{big_column_mut, column_mut, scalar_column};
#[cfg(any(test, feature = "enable-core"))]
#[cfg_attr(not(feature = "enable-core"), allow(unused_imports))]
pub(crate) use vmp::gglwe_product_digits_strided;
