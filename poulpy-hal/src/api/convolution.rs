use crate::layouts::{
    Backend, CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecRBackendMut, CnvPVecRBackendRef,
    CnvPVecROwned, PrepareHint, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
};

/// Allocates prepared convolution operands ([`CnvPVecL`](crate::layouts::CnvPVecL), [`CnvPVecR`](crate::layouts::CnvPVecR)).
///
/// ```text
/// op         cnv_pvec_left_alloc(cols, size, hint) / cnv_pvec_right_alloc(cols, size, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1; hint: the PrepareHint the destination will be written under
/// ensures    returns an owned degree-N CnvPVecL or CnvPVecR of those dimensions in the backend's prepared representation, which is opaque; its contents are unspecified
/// exact      not an arithmetic operation
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait CnvPVecAlloc<BE: Backend> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecLOwned<BE>;
    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecROwned<BE>;
}

/// Returns the byte sizes for prepared convolution operands.
///
/// ```text
/// op         bytes_of_cnv_pvec_left(cols, size, hint) / bytes_of_cnv_pvec_right(cols, size, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns the byte size of such a prepared operand, the amount take_cnv_pvec_left_scratch and its right twin carve. The hint never changes the value a prepared operand denotes, and the in-tree backends give it the same size
/// exact      not an arithmetic operation
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait CnvPVecBytesOf {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize, hint: PrepareHint) -> usize;
    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize, hint: PrepareHint) -> usize;
}

/// Bivariate convolution over `Z[X, Y] mod (X^N + 1)` where `Y = 2^{-K}`.
///
/// Provides methods to prepare left/right operands and apply the convolution.
/// See method-level documentation for the mathematical formulation.
pub trait Convolution<BE: Backend> {
    /// Returns scratch bytes required for [`cnv_prepare_left`](Convolution::cnv_prepare_left).
    ///
    /// ```text
    /// op         cnv_prepare_left_tmp_bytes(res_size, a_size)
    /// class      support
    /// mutation   none
    /// domain     res_size, a_size: the prepared operand's and the input's limb counts
    /// ensures    returns the scratch bytes cnv_prepare_left needs on those sizes
    /// exact      not an arithmetic operation
    /// test       test_convolution
    /// ```
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the left
    /// operand of a bivariate convolution.
    ///
    /// ```text
    /// op         cnv_prepare_left(res, a, mask, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a CnvPVecL with res.cols() == a.cols(); a: a dense VecZnx of the module degree; mask: the bitwise AND applied to the coefficients of a's last live limb, which is how a caller drops the bits below its working precision, -1 keeping every bit
    /// requires   scratch >= cnv_prepare_left_tmp_bytes(res.size(), a.size())
    /// ensures    res holds prep_L(a) with that mask applied, in the representation res's PrepareHint names. The representation is opaque, so the statement is on the observable: cnv_apply_dft with it is the bivariate convolution by the masked a
    /// sparse     a is a sparse-capable slot: a degree-n input, n dividing N, produces a degree-n prepared operand standing for prep_L(switch_ring_{n->N}(a)) (4.5). Implemented in PR7 (#266)
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution, test_convolution_prepare_shape_rejected
    /// ```
    fn cnv_prepare_left(
        &self,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_right`](Convolution::cnv_prepare_right).
    ///
    /// ```text
    /// op         cnv_prepare_right_tmp_bytes(res_size, a_size)
    /// class      support
    /// mutation   none
    /// domain     res_size, a_size: the prepared operand's and the input's limb counts
    /// ensures    returns the scratch bytes cnv_prepare_right needs on those sizes
    /// exact      not an arithmetic operation
    /// test       test_convolution
    /// ```
    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the right
    /// operand of a bivariate convolution.
    ///
    /// ```text
    /// op         cnv_prepare_right(res, a, mask, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a CnvPVecR with res.cols() == a.cols(); a: a dense VecZnx of the module degree; mask as for cnv_prepare_left
    /// requires   scratch >= cnv_prepare_right_tmp_bytes(res.size(), a.size())
    /// ensures    res holds prep_R(a) with that mask applied, in the representation res's PrepareHint names, observed through cnv_apply_dft
    /// sparse     a is a sparse-capable slot, as for cnv_prepare_left (4.5)
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution, test_convolution_prepare_shape_rejected
    /// ```
    fn cnv_prepare_right(
        &self,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_apply_dft`](Convolution::cnv_apply_dft).
    ///
    /// ```text
    /// op         cnv_apply_dft_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     the sizes of the destination and of the two prepared operands
    /// ensures    returns the scratch bytes cnv_apply_dft needs on those sizes
    /// exact      not an arithmetic operation
    /// test       test_convolution
    /// ```
    fn cnv_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Returns scratch bytes required for [`cnv_by_const_apply`](Convolution::cnv_by_const_apply).
    ///
    /// ```text
    /// op         cnv_by_const_apply_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     the sizes of the destination and of the two coefficient-domain operands
    /// ensures    returns the scratch bytes cnv_by_const_apply needs on those sizes
    /// exact      not an arithmetic operation
    /// test       test_convolution_by_const
    /// ```
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
    ///
    /// Required of every backend, never derived: it is an exact big-domain
    /// product of `a` with one coefficient column of `b`, and the DFT
    /// decomposition would route it through an approximate transform on FFT64.
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a VecZnxBig; a, b: dense VecZnx of the module degree; b_coeff < b.n()
    /// requires   scratch >= cnv_by_const_apply_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    res[res_col] is the bivariate convolution of a[a_col] with coefficient b_coeff of b[b_col], read as a constant in X, scaled by 2^(cnv_offset * base2k); limbs past the convolution bound are zero-filled
    /// sparse     a is the sparse-capable slot, through the switch_ring substitution of 4.5, and stays derived there
    /// exact      exact: it is a big-domain product, which is why it is required of every backend rather than routed through the lossy DFT decomposition on FFT64
    /// test       test_convolution_by_const
    /// ```
    fn cnv_by_const_apply(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_by_const_apply_add`](Convolution::cnv_by_const_apply_add).
    ///
    /// ```text
    /// op         cnv_by_const_apply_add_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     the sizes of the destination and of the two coefficient-domain operands
    /// ensures    returns the scratch bytes cnv_by_const_apply_add needs: one res_size-limb VecZnxBig plus the product's own scratch. It is not cnv_by_const_apply_tmp_bytes
    /// exact      not an arithmetic operation
    /// test       test_convolution_by_const_add
    /// ```
    fn cnv_by_const_apply_add_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// `res[res_col] +=` the [`Convolution::cnv_by_const_apply`] result; limbs
    /// the convolution would zero-fill are left untouched. Scratch requirement
    /// is
    /// [`cnv_by_const_apply_add_tmp_bytes`](Convolution::cnv_by_const_apply_add_tmp_bytes),
    /// not [`cnv_by_const_apply_tmp_bytes`](Convolution::cnv_by_const_apply_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_by_const_apply_add(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition vec_znx_big_add_assign(res, res_col, cnv_by_const_apply(cnv_offset, tmp, a, a_col, b, b_col, b_coeff), 0), tmp of res.size() limbs
    /// domain     as for cnv_by_const_apply
    /// requires   scratch >= cnv_by_const_apply_add_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    res[res_col] gains the cnv_by_const_apply result; the limbs the product would zero-fill gain zero and so keep their value
    /// fallback   OEP default body: the product into a carved res.size()-limb VecZnxBig, then vec_znx_big_add_assign
    /// override   allowed, with cnv_by_const_apply_add_tmp_bytes
    /// exact      exact, as for cnv_by_const_apply
    /// test       test_convolution_by_const_add, test_cnv_by_const_apply_add_derived
    /// ```
    fn cnv_by_const_apply_add(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
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
    ///
    /// ```text
    /// op         cnv_apply_dft(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a VecZnxDft; a: a CnvPVecL; b: a CnvPVecR, all of the module degree
    /// requires   scratch >= cnv_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    idft(res[res_col]) is the bivariate convolution of a[a_col] and b[b_col] over Z[X, Y] mod (X^N + 1), Y = 2^-base2k, scaled by 2^(cnv_offset * base2k); a res shorter than a.size() + b.size() truncates in Y, and the limbs past the convolution bound are zero-filled
    /// sparse     one of a and b may be a degree-n prepared operand, under the substitution of 4.5; there is no backend-generic body for it, so the mixed-degree sweep is a basis kernel per family (PR7, #266)
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution
    /// ```
    fn cnv_apply_dft(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_apply_dft_add`](Convolution::cnv_apply_dft_add).
    ///
    /// ```text
    /// op         cnv_apply_dft_add_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     the sizes of the destination and of the two prepared operands
    /// ensures    returns the scratch bytes cnv_apply_dft_add needs: one res_size-limb VecZnxDft plus the convolution's own scratch. It is not cnv_apply_dft_tmp_bytes
    /// exact      not an arithmetic operation
    /// test       test_convolution_add
    /// ```
    fn cnv_apply_dft_add_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Accumulating variant of [`cnv_apply_dft`](Convolution::cnv_apply_dft):
    /// `res[res_col] += a[a_col] (x) b[b_col]`, bit-identical to `cnv_apply_dft`
    /// followed by a DFT-domain add. Limbs `>= min(res.size(), a.size() + b.size())`
    /// are left untouched. Scratch requirement is
    /// [`cnv_apply_dft_add_tmp_bytes`](Convolution::cnv_apply_dft_add_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_apply_dft_add(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition vec_znx_dft_add_assign(res, res_col, cnv_apply_dft(cnv_offset, tmp, a, a_col, b, b_col), 0), tmp of res.size() limbs
    /// domain     as for cnv_apply_dft
    /// requires   scratch >= cnv_apply_dft_add_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    res[res_col] gains the cnv_apply_dft result, bit-identically to that call followed by a DFT-domain add; the limbs past min(res.size(), a.size() + b.size()) gain zero and so keep their value
    /// sparse     as for cnv_apply_dft (4.5)
    /// fallback   OEP default body: the convolution into a carved res.size()-limb VecZnxDft, then vec_znx_dft_add_assign
    /// override   allowed, with cnv_apply_dft_add_tmp_bytes
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution_add, test_cnv_apply_dft_add_derived
    /// ```
    fn cnv_apply_dft_add(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_apply_dft_sum`](Convolution::cnv_apply_dft_sum).
    ///
    /// `a_size` and `b_size` are upper bounds over the sizes of the term operands.
    ///
    /// ```text
    /// op         cnv_apply_dft_sum_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     a_size and b_size are upper bounds over the term operands' sizes
    /// ensures    returns the scratch bytes cnv_apply_dft_sum needs: the larger of the overwriting and the accumulating product the per-term fallback chains
    /// exact      not an arithmetic operation
    /// test       test_convolution_sum
    /// ```
    fn cnv_apply_dft_sum_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Evaluates a sum of bivariate convolutions: `res[res_col] = Σ_t a_t ⊛ b_t`,
    /// scaled by `2^{cnv_offset * Base2K}`, overwriting `res[res_col]`.
    ///
    /// Each term behaves like one [`Convolution::cnv_apply_dft`] call over the
    /// selected columns and the per-term results are summed; with an empty
    /// `terms` slice the output column is zeroed. Backends may fuse the
    /// accumulation (one lazy reduction per output limb, destination written
    /// once), so the result is congruent to — but not necessarily bit-identical
    /// with — a sequence of [`Convolution::cnv_apply_dft_add`] calls.
    ///
    /// ```text
    /// op         cnv_apply_dft_sum(cnv_offset, res, res_col, terms, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition cnv_apply_dft on the first term, then cnv_apply_dft_add on each of the rest
    /// domain     res: a VecZnxDft; terms: prepared left and right operands with their column indices
    /// requires   scratch >= cnv_apply_dft_sum_tmp_bytes(cnv_offset, res.size(), a_size, b_size)
    /// ensures    idft(res[res_col]) is the sum over the terms of their bivariate convolutions, overwriting the column; an empty slice zeroes it. A backend may fuse the accumulation with one lazy reduction per output limb, so the result is congruent to a chain of cnv_apply_dft_add calls without being bit-identical to it
    /// sparse     per term, as for cnv_apply_dft (4.5)
    /// fallback   OEP default body: the first term overwrites with cnv_apply_dft, which also zeroes the limbs past the convolution bound, and the remaining terms fold in with cnv_apply_dft_add
    /// override   allowed, with cnv_apply_dft_sum_tmp_bytes; the avx and avx512 families override it with a fused kernel
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution_sum, test_cnv_apply_dft_sum_derived
    /// ```
    fn cnv_apply_dft_sum<'a>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        terms: &[CnvDftAccTerm<'a, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: 'a;

    /// Returns scratch bytes required for [`cnv_pairwise_apply_dft`](Convolution::cnv_pairwise_apply_dft).
    ///
    /// ```text
    /// op         cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, res_size, a_size, b_size)
    /// class      support
    /// mutation   none
    /// domain     the sizes of the destination and of the two prepared operands
    /// ensures    returns the scratch bytes cnv_pairwise_apply_dft needs: the larger of the overwriting and the accumulating product it chains
    /// exact      not an arithmetic operation
    /// test       test_convolution_pairwise
    /// ```
    fn cnv_pairwise_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates the bivariate pair-wise convolution res = (a\[i\] + a\[j\]) * (b\[i\] + b\[j\]).
    /// If i == j then calls [Convolution::cnv_apply_dft], i.e. res = a\[i\] * b\[i\].
    /// See [Convolution::cnv_apply_dft] for information about the bivariate convolution.
    ///
    /// ```text
    /// op         cnv_pairwise_apply_dft(cnv_offset, res, res_col, a, b, i, j, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition cnv_apply_dft(a[i], b[i]) then cnv_apply_dft_add of (a[i], b[j]), (a[j], b[i]) and (a[j], b[j]); i == j degenerates to the single product
    /// domain     res: a VecZnxDft; a: a CnvPVecL; b: a CnvPVecR; i, j column indices
    /// requires   scratch >= cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    idft(res[res_col]) is the convolution of (a[i] + a[j]) with (b[i] + b[j]), expanded in the DFT domain where the prepared operands are linear
    /// sparse     per product, as for cnv_apply_dft (4.5)
    /// fallback   OEP default body: the four-product expansion above
    /// override   allowed, with cnv_pairwise_apply_dft_tmp_bytes; the avx512 family overrides it with a fused kernel
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_convolution_pairwise, test_cnv_pairwise_apply_dft_derived
    /// ```
    fn cnv_pairwise_apply_dft(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_self`](Convolution::cnv_prepare_self).
    ///
    /// ```text
    /// op         cnv_prepare_self_tmp_bytes(res_size, a_size)
    /// class      support
    /// mutation   none
    /// domain     res_size, a_size: the prepared operands' and the input's limb counts
    /// ensures    returns the scratch bytes cnv_prepare_self needs: the larger of the two prepares
    /// exact      not an arithmetic operation
    /// test       test_cnv_prepare_self_derived
    /// ```
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;

    /// Prepares both left and right convolution operands from the same input polynomial,
    /// sharing the FFT/NTT computation. This is an optimization for self-convolution
    /// (squaring) where both operands are the same polynomial.
    ///
    /// ```text
    /// op         cnv_prepare_self(left, right, a, mask, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition cnv_prepare_left(left, a, mask) and cnv_prepare_right(right, a, mask)
    /// domain     left: a CnvPVecL; right: a CnvPVecR with right.cols() == left.cols() and right.size() == left.size(); a: a dense VecZnx of the module degree with a.cols() == left.cols(); mask as for cnv_prepare_left
    /// requires   scratch >= cnv_prepare_self_tmp_bytes(left.size(), a.size())
    /// ensures    left holds prep_L(a) and right holds prep_R(a), the pair a self-convolution needs
    /// sparse     a is a sparse-capable slot, as for cnv_prepare_left (4.5)
    /// fallback   OEP default body: the two prepares in sequence
    /// override   allowed, with cnv_prepare_self_tmp_bytes; a backend that shares the transform between the two does it here
    /// exact      backend DFT class: exact for the NTT families, approximate for FFT64
    /// test       test_cnv_prepare_self_derived, test_convolution_prepare_shape_rejected
    /// ```
    fn cnv_prepare_self(
        &self,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
