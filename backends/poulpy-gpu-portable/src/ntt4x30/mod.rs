//! DFT-domain kernels of the NTT4x30 family.
//!
//! A DFT word is the four residues of a coefficient modulo `Primes30`. Inside
//! a `(limb, column)` block of `n` words the residues are stored in four
//! planes of `n` `u32`, one per prime, canonical in `[0, q)`: the element-wise
//! operations are plain modular additions, and a transform works on one plane
//! at a time.
//!
//! Over the conjugate-invariant ring the transform is the negacyclic NTT of
//! the same degree behind a basis change: `c -> b` with
//! `b_0 = c_0`, `b_j = alpha^-j (c_j - i c_{n-j})`, `alpha` a primitive `4n`-th
//! root and `i = alpha^n`. Slot `s` then holds the value of the ambient
//! degree-`2n` polynomial at `alpha^(4 bitrev(s) + 1)`.

mod automorphism;
mod cnv;
mod crt;
mod elementwise;
mod ntt;
mod primes;
mod svp;
pub(crate) mod tables;
mod vmp;

pub use automorphism::AutomorphismPlan;
pub(crate) use automorphism::automorphism;
pub(crate) use cnv::{CAPACITY_STEP, convolution, convolution_pairwise, convolution_sum, convolution_tensor};
#[cfg_attr(not(feature = "enable-core"), allow(unused_imports))]
pub(crate) use cnv::{MANY_LEFTS, MANY_PRODUCTS, MANY_TERMS, Schedule, scale_many};
pub(crate) use crt::{crt_at, to_big, to_big_clearing};
pub(crate) use elementwise::{DftTerm, combine};
pub(crate) use ntt::{forward, inverse};
pub(crate) use primes::Consts;
pub(crate) use svp::scale;
pub(crate) use tables::Tables;
pub(crate) use vmp::{Pmat, Tail, digit_product, extract_rows};

use crate::runtime::Binding;

/// One column of a DFT-domain vector as a kernel addresses it: plane `k` of
/// limb `j` starts at `u32` word `base + j * stride + k * n` of `binding`.
pub(crate) struct DftColumn {
    pub(crate) binding: Binding,
    pub(crate) base: u32,
    pub(crate) stride: u32,
    /// Coefficients per limb.
    pub(crate) n: usize,
    /// Limbs.
    pub(crate) size: usize,
}

impl DftColumn {
    /// The `count` limbs `first, first + step, ..` of the column, renumbered from zero.
    pub(crate) fn limbs(&self, first: usize, step: usize, count: usize) -> Self {
        let narrow = |words: usize| u32::try_from(words).expect("DFT view beyond the 32-bit word range");
        let (base, stride) = if count == 0 {
            (self.base, self.stride)
        } else {
            assert!(first + (count - 1) * step < self.size, "DFT limb selection out of range");
            (
                narrow(self.base as usize + first * self.stride as usize),
                narrow(step * self.stride as usize),
            )
        };
        Self {
            binding: self.binding.clone(),
            base,
            stride,
            n: self.n,
            size: count,
        }
    }
}
