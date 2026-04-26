use crate::{
    layouts::{Backend, NoiseInfos, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

/// Zeroes all limbs of the selected column.
pub trait VecZnxZero {
    fn vec_znx_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut;
}

pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNormalizeAssign<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_assign<A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAddInto {
    /// Adds the selected column of `a` to the selected column of `b` and writes the result on the selected column of `res`.
    fn vec_znx_add_into<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxAddAssign {
    /// Adds the selected column of `a` to the selected column of `res` and writes the result on the selected column of `res`.
    fn vec_znx_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAddScalarInto {
    /// Adds the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into<R, A, B>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxAddScalarAssign {
    /// Adds the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_add_scalar_assign<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxSub {
    /// Subtracts the selected column of `b` from the selected column of `a` and writes the result on the selected column of `res`.
    fn vec_znx_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxSubAssign {
    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res\[res_col\] -= a\[a_col\]
    fn vec_znx_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubNegateAssign {
    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res\[res_col\] = a\[a_col\] - res\[res_col\]
    fn vec_znx_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubScalar {
    /// Subtracts the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxSubScalarAssign {
    /// Subtracts the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_sub_scalar_assign<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxNegate {
    /// Negates the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNegateAssign {
    /// Negates the selected column of `a`.
    fn vec_znx_negate_assign<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// Returns scratch bytes required for left-shift operations.
pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxLsh<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshAddInto<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// Returns scratch bytes required for right-shift operations.
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRsh<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRshAddInto<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshSub<B: Backend> {
    /// Left shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRshSub<B: Backend> {
    /// Right shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshAssign<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    fn vec_znx_lsh_assign<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRshAssign<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    fn vec_znx_rsh_assign<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRotate {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateAssign<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_assign<A>(&self, p: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAutomorphism {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_assign<R>(&self, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut;
}

/// Multiplies the selected column by `(X^p - 1)` in `Z[X]/(X^N + 1)`.
pub trait VecZnxMulXpMinusOne {
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMulXpMinusOneAssign<B: Backend> {
    fn vec_znx_mul_xp_minus_one_assign<R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut;
}

pub trait VecZnxSplitRingTmpBytes {
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize;
}

pub trait VecZnxSplitRing<B: Backend> {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split_ring<R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMergeRingsTmpBytes {
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMergeRings<B: Backend> {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge_rings<R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// Switches ring degree between `a` and `res` by truncation or zero-padding.
pub trait VecZnxSwitchRing {
    fn vec_znx_switch_ring<R, A>(&self, res: &mut R, res_col: usize, a: &A, col_a: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// Copies the selected column of `a` into the selected column of `res`.
pub trait VecZnxCopy {
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{base2k-1}, 2^{base2k-1}\]
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
/// Fills the selected column with a discrete Gaussian noise vector
/// scaled by `2^{-k}` with standard deviation `sigma`, bounded to `[-bound, bound]`.
pub trait VecZnxFillNormal {
    fn vec_znx_fill_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormal {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn vec_znx_add_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut;
}
