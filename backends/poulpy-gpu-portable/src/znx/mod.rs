//! Coefficient-domain kernels, one thread per coefficient (one per limb for
//! the sum over the coefficients of a limb). They serve both
//! `i64` limbs and `i128` big values: every buffer is addressed in 64-bit
//! words, a big value being its low word followed by its high word.

mod gadget;
mod lincomb;
mod normalize;
mod permute;
mod products;

pub(crate) use gadget::gadget_decompose;
pub(crate) use lincomb::{Term, lincomb, lincomb_decimating};
pub(crate) use normalize::{KERNEL_RADIX, normalize, normalize_residues};
pub(crate) use permute::{Map, ci_embed, ci_trace, move_from_temp, permute};
pub(crate) use products::{col_weighted_sum, convolution_by_const, inner_sum, scalar_product};

use crate::runtime::Binding;

/// One column of a coefficient-domain vector as a kernel addresses it:
/// coefficient `i` of limb `j` starts at 64-bit word `base + j * stride + i * words`
/// of `binding`.
pub(crate) struct Column {
    pub(crate) binding: Binding,
    pub(crate) base: u32,
    pub(crate) stride: u32,
    /// 64-bit words per coefficient: 1 for `i64`, 2 for `i128`.
    pub(crate) words: u32,
    /// Visible coefficients per limb.
    pub(crate) n: usize,
    /// Visible limbs.
    pub(crate) size: usize,
}
