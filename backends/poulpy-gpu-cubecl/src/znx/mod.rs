//! Coefficient-domain kernels: `i64` limbs, one thread per coefficient.

mod gadget;
mod lincomb;
mod normalize;
mod permute;

pub(crate) use gadget::gadget_decompose;
pub(crate) use lincomb::{Term, lincomb, lincomb_decimating};
pub(crate) use normalize::normalize;
pub(crate) use permute::{Map, ci_embed, ci_trace, permute};

use crate::runtime::Binding;

/// One column of a coefficient-domain vector as a kernel addresses it:
/// coefficient `i` of limb `j` is element `base + j * stride + i` of `binding`.
pub(crate) struct Column {
    pub(crate) binding: Binding,
    pub(crate) base: u32,
    pub(crate) stride: u32,
    /// Visible coefficients per limb.
    pub(crate) n: usize,
    /// Visible limbs.
    pub(crate) size: usize,
}

/// Threads per group of the coefficient-domain kernels.
pub(crate) const DIM: u32 = 256;
