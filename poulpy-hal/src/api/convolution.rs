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
/// ensures    returns an owned CnvPVecL or CnvPVecR of the module degree and those dimensions in the backend's prepared representation, which is opaque; its contents are unspecified
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
/// ensures    returns the byte size of such a prepared operand, the amount take_cnv_pvec_left_scratch and its right twin carve. The hint never changes the value a prepared operand denotes, and every backend gives it the same size
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
    /// test       test_convolution
    /// ```
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the left
    /// operand of a bivariate convolution.
    ///
    /// ```text
    /// op         cnv_prepare_left(res, a, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a CnvPVecL of the module degree with res.cols() == a.cols(); a: a dense VecZnx of the module degree, canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_left_tmp_bytes(res.size(), a.size())
    /// ensures    res holds prep_L(a) in the representation res's PrepareHint names, observed through cnv_apply_dft
    /// test       test_convolution, test_convolution_prepare_shape_rejected, test_convolution_sparse
    /// ```
    fn cnv_prepare_left(
        &self,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
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
    /// test       test_convolution
    /// ```
    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the right
    /// operand of a bivariate convolution.
    ///
    /// ```text
    /// op         cnv_prepare_right(res, a, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a CnvPVecR of the module degree with res.cols() == a.cols(); a: a dense VecZnx of the module degree, canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_right_tmp_bytes(res.size(), a.size())
    /// ensures    res holds prep_R(a) in the representation res's PrepareHint names, observed through cnv_apply_dft
    /// test       test_convolution, test_convolution_prepare_shape_rejected, test_convolution_sparse
    /// ```
    fn cnv_prepare_right(
        &self,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
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
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a VecZnxBig of the module degree; a: a dense VecZnx of the module degree; b: a dense VecZnx of any degree, b_coeff < b.n()
    /// requires   scratch >= cnv_by_const_apply_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    res[res_col] is the bivariate convolution of a[a_col] with coefficient b_coeff of b[b_col], read as a constant in X, scaled by 2^(cnv_offset * base2k); limbs past the convolution bound are zero-filled
    /// test       test_convolution_by_const, test_convolution_by_const_degree_rejected
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
    /// test       test_convolution_by_const_add, test_cnv_by_const_apply_add_derived, test_convolution_by_const_degree_rejected
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
    /// A res with fewer than a.size() + b.size() - 1 - cnv_offset limbs truncates the result in Y; the limbs past that bound are zero-filled.
    ///
    /// ```text
    /// op         cnv_apply_dft(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of the module degree; b: a CnvPVecR of the module degree or of a degree dividing it
    /// requires   scratch >= cnv_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    idft(res[res_col]) is the bivariate convolution of a[a_col] and b[b_col] over Z[X, Y] mod (X^N + 1), Y = 2^-base2k, scaled by 2^(cnv_offset * base2k); a res with fewer than a.size() + b.size() - 1 - cnv_offset limbs truncates in Y, and the limbs past that bound are zero-filled
    /// sparse     b may be a prepared right operand of degree n, n a power of two dividing N and not below the backend's minimum sparse degree, prepared under a degree-n module; res and a take the module degree; the degree embedding of the api module docs defines the correspondence that reads it
    /// test       test_convolution, test_convolution_sparse
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
    /// test       test_convolution_add
    /// ```
    fn cnv_apply_dft_add_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Accumulating variant of [`cnv_apply_dft`](Convolution::cnv_apply_dft):
    /// `res[res_col] += a[a_col] (x) b[b_col]`, bit-identical to `cnv_apply_dft`
    /// followed by a DFT-domain add. Limbs past the convolution bound are left
    /// untouched. Scratch requirement is
    /// [`cnv_apply_dft_add_tmp_bytes`](Convolution::cnv_apply_dft_add_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_apply_dft_add(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition vec_znx_dft_add_assign(res, res_col, cnv_apply_dft(cnv_offset, tmp, a, a_col, b, b_col), 0), tmp of res.size() limbs
    /// domain     as for cnv_apply_dft
    /// requires   scratch >= cnv_apply_dft_add_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    res[res_col] gains the cnv_apply_dft result, bit-identically to that call followed by a DFT-domain add; the limbs past the convolution bound gain zero and so keep their value
    /// sparse     as for cnv_apply_dft
    /// fallback   OEP default body: the convolution into a carved res.size()-limb VecZnxDft, then vec_znx_dft_add_assign
    /// override   allowed, with cnv_apply_dft_add_tmp_bytes
    /// test       test_convolution_add, test_cnv_apply_dft_add_derived, test_convolution_sparse
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
    /// test       test_convolution_sum
    /// ```
    fn cnv_apply_dft_sum_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Evaluates a sum of bivariate convolutions: `res[res_col] = Σ_t a_t ⊛ b_t`,
    /// scaled by `2^{cnv_offset * Base2K}`, overwriting `res[res_col]`.
    ///
    /// Each term behaves like one [`Convolution::cnv_apply_dft`] call over the
    /// selected columns and the per-term results are summed; with an empty
    /// `terms` slice the output column is zeroed. A backend may fuse the
    /// accumulation (one lazy reduction per output limb, the destination written
    /// once); the DFT-domain bytes may then differ from a sequence of
    /// [`Convolution::cnv_apply_dft_add`] calls, idft of the result is the same.
    ///
    /// ```text
    /// op         cnv_apply_dft_sum(cnv_offset, res, res_col, terms, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition cnv_apply_dft on the first term, then cnv_apply_dft_add on each of the rest
    /// domain     res: a VecZnxDft of the module degree; terms: prepared left operands of the module degree and right operands of the module degree or of a degree dividing it, with their column indices
    /// requires   scratch >= cnv_apply_dft_sum_tmp_bytes(cnv_offset, res.size(), a_size, b_size)
    /// ensures    idft(res[res_col]) is the sum over the terms of their bivariate convolutions, overwriting the column; an empty slice zeroes it. A backend may fuse the accumulation with one lazy reduction per output limb, so the DFT-domain bytes may differ from a chain of cnv_apply_dft_add calls while idft of the result is the same
    /// sparse     per term, as for cnv_apply_dft
    /// fallback   OEP default body: the first term overwrites with cnv_apply_dft, which also zeroes the limbs past the convolution bound, and the remaining terms fold in with cnv_apply_dft_add
    /// override   allowed, with cnv_apply_dft_sum_tmp_bytes
    /// test       test_convolution_sum, test_cnv_apply_dft_sum_derived, test_convolution_sparse
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
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of the module degree; b: a CnvPVecR of the module degree or of a degree dividing it; i, j column indices
    /// requires   scratch >= cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    for i != j, idft(res[res_col]) is the convolution of (a[i] + a[j]) with (b[i] + b[j]), expanded in the DFT domain where the prepared operands are linear; for i == j it is the single product a[i] * b[i], not the four-fold one the sum would give
    /// sparse     per product, as for cnv_apply_dft
    /// fallback   OEP default body: the four-product expansion above
    /// override   allowed, with cnv_pairwise_apply_dft_tmp_bytes
    /// test       test_convolution_pairwise, test_cnv_pairwise_apply_dft_derived, test_convolution_sparse
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
    /// test       test_cnv_prepare_self_derived
    /// ```
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;

    /// Prepares both left and right convolution operands from the same input polynomial,
    /// so a backend can share the transform between the two.
    ///
    /// ```text
    /// op         cnv_prepare_self(left, right, a, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition cnv_prepare_left(left, a) and cnv_prepare_right(right, a)
    /// domain     left: a CnvPVecL of the module degree; right: a CnvPVecR of the module degree, with right.cols() == left.cols() and right.size() == left.size(); a: a dense VecZnx of the module degree with a.cols() == left.cols(), canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_self_tmp_bytes(left.size(), a.size())
    /// ensures    left holds prep_L(a) and right holds prep_R(a), the pair a self-convolution needs
    /// fallback   OEP default body: the two prepares in sequence
    /// override   allowed, with cnv_prepare_self_tmp_bytes; a backend that shares the transform between the two does it here
    /// test       test_cnv_prepare_self_derived, test_convolution_prepare_shape_rejected, test_convolution_sparse
    /// ```
    fn cnv_prepare_self(
        &self,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
