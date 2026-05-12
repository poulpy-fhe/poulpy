use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef, ScratchArena,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
};

/// Allocates prepared convolution operands ([`CnvPVecL`], [`CnvPVecR`]).
pub trait CnvPVecAlloc<BE: Backend> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecL<BE::OwnedBuf, BE>;
    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecR<BE::OwnedBuf, BE>;
}

/// Returns the byte sizes for prepared convolution operands.
pub trait CnvPVecBytesOf {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize;
}

/// Bivariate convolution over `Z[X, Y] mod (X^N + 1)` where `Y = 2^{-K}`.
///
/// Provides methods to prepare left/right operands and apply the convolution.
/// See method-level documentation for the mathematical formulation.
pub trait Convolution<BE: Backend> {
    /// Returns scratch bytes required for [`cnv_prepare_left`](Convolution::cnv_prepare_left).
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the left
    /// operand of a bivariate convolution.
    fn cnv_prepare_left<'s, 'r>(
        &self,
        res: &mut CnvPVecLBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_right`](Convolution::cnv_prepare_right).
    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the right
    /// operand of a bivariate convolution.
    fn cnv_prepare_right<'s, 'r>(
        &self,
        res: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );

    /// Returns scratch bytes required for [`cnv_apply_dft`](Convolution::cnv_apply_dft).
    fn cnv_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Returns scratch bytes required for [`cnv_by_const_apply`](Convolution::cnv_by_const_apply).
    fn cnv_by_const_apply_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Evaluates a bivariate convolution over Z\[X, Y\] (x) Z\[Y\] mod (X^N + 1) where Y = 2^-K over the
    /// selected columns and stores the result on the selected column, scaled by 2^{cnv_offset * Base2K}
    ///
    /// Behavior is identical to [Convolution::cnv_apply_dft] with `b` treated as a constant polynomial
    /// in the X variable, for example:
    ///```text
    ///       1    X   X^2  X^3
    /// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     Y [a01, a11, a21, a31]
    ///
    /// b = 1 [b0] = (b00 + b01 * 2^-K)
    ///     Y [b0]
    /// ```
    /// This method is intended to be used for multiplications by constants that are greater than the base2k.
    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    /// Evaluates a bivariate convolution over Z\[X, Y\] (x) Z\[X, Y\] mod (X^N + 1) where Y = 2^-K over the
    /// selected columns and stores the result on the selected column, scaled by 2^{cnv_offset * Base2K}
    ///
    /// # Example
    ///```text
    ///       1    X   X^2  X^3
    /// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     Y [a01, a11, a21, a31]
    ///
    /// b = 1 [b00, b10, b20, b30] = (b00 + b01 * 2^-K) + (b10 + b11 * 2^-K) * X ...
    ///     Y [b01, b11, b21, b31]
    ///
    /// If cnv_offset = 0:
    ///
    ///            1    X   X^2  X^3
    /// res = 1  [r00, r10, r20, r30] = (r00 + r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K) + ... * X + ...
    ///       Y  [r01, r11, r21, r31]
    ///       Y^2[r02, r12, r22, r32]
    ///       Y^3[r03, r13, r23, r33]
    ///
    /// If cnv_offset = 1:
    ///
    ///            1    X   X^2  X^3
    /// res = 1  [r01, r11, r21, r31]  = (r01 + r02 * 2^-K + r03 * 2^-2K) + ... * X + ...
    ///       Y  [r02, r12, r22, r32]
    ///       Y^2[r03, r13, r23, r33]
    ///       Y^3[  0,   0,   0 ,  0]
    /// ```
    /// If res.size() < a.size() + b.size() + k, result is truncated accordingly in the Y dimension.
    fn cnv_apply_dft<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    /// Returns scratch bytes required for [`cnv_pairwise_apply_dft`](Convolution::cnv_pairwise_apply_dft).
    fn cnv_pairwise_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates the bivariate pair-wise convolution res = (a\[i\] + a\[j\]) * (b\[i\] + b\[j\]).
    /// If i == j then calls [Convolution::cnv_apply_dft], i.e. res = a\[i\] * b\[i\].
    /// See [Convolution::cnv_apply_dft] for information about the bivariate convolution.
    fn cnv_pairwise_apply_dft<'s>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_self`](Convolution::cnv_prepare_self).
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;

    /// Prepares both left and right convolution operands from the same input polynomial,
    /// sharing the FFT/NTT computation. This is an optimization for self-convolution
    /// (squaring) where both operands are the same polynomial.
    fn cnv_prepare_self<'s, 'l, 'r>(
        &self,
        left: &mut CnvPVecLBackendMut<'l, BE>,
        right: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );
}
