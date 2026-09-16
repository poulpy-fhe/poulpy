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
/// ensures    returns the byte size required for the given prepared operand dimensions and hint
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
    /// definition res[c,j] reads as a[c,j] for every 0 <= c < res.cols()
    /// domain     res: a CnvPVecL of the module degree with res.cols() == a.cols(); a: a dense VecZnx of the module degree, canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_left_tmp_bytes(res.size(), a.size())
    /// ensures    each prepared column reads the source column truncated or zero-extended to res.size() limbs
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
    /// definition res[c,j] reads as a[c,j] for every 0 <= c < res.cols()
    /// domain     res: a CnvPVecR of the module degree with res.cols() == a.cols(); a: a dense VecZnx of the module degree, canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_right_tmp_bytes(res.size(), a.size())
    /// ensures    each prepared column reads the source column truncated or zero-extended to res.size() limbs
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

    /// Convolves the selected column of `a` with coefficient `b_coeff` of
    /// each limb of `b`, treating that coefficient as a constant polynomial.
    /// Output limb `j` sums products whose input limb indices total
    /// `j + cnv_offset`. With the value model's limb weights, the selected
    /// product window has scale `2^((cnv_offset + 1) * w)` at radix width `w`.
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition res[res_col,j] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v,b_coeff]; other columns of res are unchanged
    /// domain     res: a VecZnxBig of the module degree; a: a dense VecZnx of the module degree; b: a dense VecZnx of any degree, b_coeff < b.n(); res.size(), a.size() and b.size() >= 1
    /// requires   scratch >= cnv_by_const_apply_tmp_bytes(cnv_offset, res.size(), a.size(), b.size()); every coefficient of each partial sum and result is representable in BigWord
    /// ensures    the selected column holds the limb window of the constant convolution, scaled by 2^((cnv_offset + 1) * w) at any radix width w; the remaining limbs are zero-filled
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
    /// ensures    returns the scratch bytes required by cnv_by_const_apply_add for the given offset and sizes
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
    /// definition res[res_col,j] = old(res)[res_col,j] + sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v,b_coeff]; selected-column limbs for which the sum index set is empty and other columns of res are unchanged
    /// domain     as for cnv_by_const_apply
    /// requires   scratch >= cnv_by_const_apply_add_tmp_bytes(cnv_offset, res.size(), a.size(), b.size()); every coefficient of each partial sum and result is representable in BigWord
    /// ensures    the selected column gains the constant convolution; limbs outside its support retain their old value
    /// fallback   the product into a one-column res.size()-limb VecZnxBig, then vec_znx_big_add_assign
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
    /// Convolves the selected prepared columns in the polynomial ring.
    /// Output limb `j` sums products whose input limb indices total
    /// `j + cnv_offset`. With the value model's limb weights, the selected
    /// product window has scale `2^((cnv_offset + 1) * w)` at radix width `w`.
    /// A shorter destination truncates the window; remaining limbs are zero.
    ///
    /// ```text
    /// op         cnv_apply_dft(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition idft(res)[res_col,j] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v]; other columns of res are unchanged
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of the module degree; b: a CnvPVecR of the module degree or of a degree dividing it; res.size(), a.size() and b.size() >= 1
    /// requires   scratch >= cnv_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    the selected inverse column holds the limb window of the convolution, scaled by 2^((cnv_offset + 1) * w) at any radix width w; the remaining limbs are zero-filled
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
    /// ensures    returns the scratch bytes required by cnv_apply_dft_add for the given offset and sizes
    /// test       test_convolution_add
    /// ```
    fn cnv_apply_dft_add_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Accumulating variant of [`cnv_apply_dft`](Convolution::cnv_apply_dft):
    /// adds the selected convolution to the old destination column.
    /// Limbs outside the convolution's support retain their old value.
    /// Scratch requirement is
    /// [`cnv_apply_dft_add_tmp_bytes`](Convolution::cnv_apply_dft_add_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         cnv_apply_dft_add(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition idft(res)[res_col,j] = idft(old(res))[res_col,j] + sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v]; selected-column limbs for which the sum index set is empty and other columns of res are unchanged
    /// domain     as for cnv_apply_dft
    /// requires   scratch >= cnv_apply_dft_add_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    the selected column gains the convolution; limbs outside its support retain their old value
    /// sparse     as for cnv_apply_dft
    /// fallback   the convolution into a one-column res.size()-limb VecZnxDft, then vec_znx_dft_add_assign
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
    /// ensures    returns the scratch bytes cnv_apply_dft_sum needs on those sizes
    /// test       test_convolution_sum
    /// ```
    fn cnv_apply_dft_sum_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Sums the selected convolution limb windows into `res[res_col]`,
    /// using the same `cnv_offset` for every term.
    ///
    /// Each term behaves like one [`Convolution::cnv_apply_dft`] call over the
    /// selected columns and the per-term results are summed; with an empty
    /// `terms` slice the output column is zeroed. Backends may fuse the
    /// accumulation while preserving the inverse-transform observable of
    /// the sum within the module's DFT exactness class.
    ///
    /// ```text
    /// op         cnv_apply_dft_sum(cnv_offset, res, res_col, terms, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition idft(res)[res_col,j] = sum_{0 <= t < terms.len()} sum_{0 <= u < terms[t].a.size(), 0 <= v < terms[t].b.size(), u + v = j + cnv_offset} terms[t].a[terms[t].a_col,u] * terms[t].b[terms[t].b_col,v]; an empty terms slice yields zero; other columns of res are unchanged
    /// domain     res: a VecZnxDft of the module degree; terms: prepared left operands of the module degree and right operands of the module degree or of a degree dividing it, with their column indices
    /// requires   scratch >= cnv_apply_dft_sum_tmp_bytes(cnv_offset, res.size(), max({0} union {terms[t].a.size() : 0 <= t < terms.len()}), max({0} union {terms[t].b.size() : 0 <= t < terms.len()}))
    /// ensures    the selected inverse column is the sum of the selected convolution limb windows; an empty slice zeroes the column
    /// sparse     per term, as for cnv_apply_dft
    /// fallback   an empty slice zeroes the selected column; otherwise cnv_apply_dft overwrites with the first term and cnv_apply_dft_add accumulates the remaining terms
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
    /// ensures    returns the scratch bytes cnv_pairwise_apply_dft needs on those sizes
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
    /// definition for every 0 <= ell < res.size(), idft(res)[res_col,ell] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = ell + cnv_offset} a[i,u] * b[i,v] if i == j, and idft(res)[res_col,ell] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = ell + cnv_offset} (a[i,u] + a[j,u]) * (b[i,v] + b[j,v]) if i != j; other columns of res are unchanged
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of the module degree; b: a CnvPVecR of the module degree or of a degree dividing it; i, j column indices
    /// requires   scratch >= cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    the selected inverse column contains one convolution when i == j and the convolution of the two column sums when i != j
    /// sparse     per product, as for cnv_apply_dft
    /// fallback   cnv_apply_dft for columns (i,i), followed when i != j by cnv_apply_dft_add for (i,j), (j,i) and (j,j)
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
    /// ensures    returns the scratch bytes cnv_prepare_self needs on those sizes
    /// test       test_cnv_prepare_self_derived
    /// ```
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;

    /// Prepares both left and right convolution operands from the same input.
    /// Implementations may share the transform between the two preparations.
    ///
    /// ```text
    /// op         cnv_prepare_self(left, right, a, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition left[c,j] and right[c,j] read as a[c,j] for every 0 <= c < left.cols() and 0 <= j < left.size()
    /// domain     left: a CnvPVecL of the module degree; right: a CnvPVecR of the module degree, with right.cols() == left.cols() and right.size() == left.size(); a: a dense VecZnx of the module degree with a.cols() == left.cols(), canonical at the precision the caller means to convolve at
    /// requires   scratch >= cnv_prepare_self_tmp_bytes(left.size(), a.size())
    /// ensures    both prepared operands read the source truncated or zero-extended to their limb count
    /// fallback   OEP default body: the shape check, then the two prepares in sequence
    /// override   allowed, with cnv_prepare_self_tmp_bytes
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
