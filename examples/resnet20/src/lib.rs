//! ResNet-20 on CIFAR-10 with poulpy-ckks and S2C-first bootstrapping.
//!
//! Every convolution runs in the last factor of the SlotsToCoeffs of a bootstrap: see [`CoeffLayout`] for where a tensor
//! sits in the coefficients and [`MergedConv`] for the convolution.

mod kernel;
mod layout;
mod merged;
pub mod model;
pub mod network;
#[cfg(test)]
mod pipeline_tests;
mod reference;
mod relu;

pub use layout::CoeffLayout;
pub use merged::{MergedConv, slots_to_blocks};
pub use relu::Relu;
