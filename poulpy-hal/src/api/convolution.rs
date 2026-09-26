use crate::layouts::{
    Backend, CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecRBackendMut, CnvPVecRBackendRef,
    CnvPVecROwned, PrepareHint, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
};

/// Allocates prepared convolution operands ([`CnvPVecL`](crate::layouts::CnvPVecL), [`CnvPVecR`](crate::layouts::CnvPVecR)).
///
/// ```text
/// op         cnv_pvec_left_alloc(n, cols, size, hint) / cnv_pvec_right_alloc(n, cols, size, hint)
/// class      support
/// mutation   none
/// domain     n: a power of two, MIN_DEGREE <= n <= the module's degree; cols >= 1, size >= 1; hint: the PrepareHint the destination will be written under
/// ensures    returns an owned degree-n CnvPVecL or CnvPVecR of those dimensions in the backend's prepared representation, which is opaque; its contents are unspecified
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait CnvPVecAlloc<BE: Backend> {
    /// Returns an owned degree-`n` [`CnvPVecL`](crate::layouts::CnvPVecL) of `cols` columns and `size` limbs under `hint`.
    fn cnv_pvec_left_alloc(&self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecLOwned<BE>;
    /// Returns an owned degree-`n` [`CnvPVecR`](crate::layouts::CnvPVecR) of `cols` columns and `size` limbs under `hint`.
    fn cnv_pvec_right_alloc(&self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecROwned<BE>;
}

/// Returns the byte sizes for prepared convolution operands.
///
/// ```text
/// op         bytes_of_cnv_pvec_left(n, cols, size, hint) / bytes_of_cnv_pvec_right(n, cols, size, hint)
/// class      support
/// mutation   none
/// domain     n: a power of two, MIN_DEGREE <= n <= the module's degree; cols >= 1, size >= 1
/// ensures    returns the byte size required for the given prepared operand dimensions and hint
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait CnvPVecBytesOf {
    /// Returns the bytes a degree-`n` [`CnvPVecL`](crate::layouts::CnvPVecL) of `cols` columns and `size` limbs under `hint` occupies.
    fn bytes_of_cnv_pvec_left(&self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> usize;
    /// Returns the bytes a degree-`n` [`CnvPVecR`](crate::layouts::CnvPVecR) of `cols` columns and `size` limbs under `hint` occupies.
    fn bytes_of_cnv_pvec_right(&self, n: usize, cols: usize, size: usize, hint: PrepareHint) -> usize;
}

/// Bivariate convolution over `Z[X, Y] mod (X^N + 1)` where `Y = 2^{-K}`.
pub trait Convolution<BE: Backend> {
    /// Returns the scratch bytes [`cnv_prepare_left`](Convolution::cnv_prepare_left) requires for those sizes.
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
    /// Writes each column of `a` into `res`, truncated or zero-extended to `res.size()` limbs.
    ///
    /// ```text
    /// op         cnv_prepare_left(res, a, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition res[c,j] reads as a[c,j] for every 0 <= c < res.cols()
    /// domain     res: a CnvPVecL of degree N with res.cols() == a.cols(); a: a dense VecZnx of degree N, canonical at the precision the caller means to convolve at
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

    /// Returns the scratch bytes [`cnv_prepare_right`](Convolution::cnv_prepare_right) requires for those sizes.
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
    /// Writes each column of `a` into `res`, truncated or zero-extended to `res.size()` limbs.
    ///
    /// ```text
    /// op         cnv_prepare_right(res, a, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition res[c,j] reads as a[c,j] for every 0 <= c < res.cols()
    /// domain     res: a CnvPVecR of degree N with res.cols() == a.cols(); a: a dense VecZnx of degree N, canonical at the precision the caller means to convolve at
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

    /// Returns the scratch bytes [`cnv_apply_dft`](Convolution::cnv_apply_dft) requires for those sizes.
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

    /// Returns the scratch bytes [`cnv_by_const_apply`](Convolution::cnv_by_const_apply) requires for those sizes.
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

    /// Writes the convolution of `a[a_col]` with coefficient `b_coeff` of `b[b_col]` into `res[res_col]`, zero-filling the remaining limbs.
    ///
    /// ```text
    /// op         cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition res[res_col,j] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v,b_coeff]; other columns of res are unchanged
    /// domain     res: a VecZnxBig of degree N; a: a dense VecZnx of degree N; b: a dense VecZnx of any degree, b_coeff < b.n(); res.size(), a.size() and b.size() >= 1
    /// requires   scratch >= cnv_by_const_apply_tmp_bytes(cnv_offset, res.size(), a.size(), b.size()); every coefficient of each partial sum and result is representable in BigWord
    /// ensures    the selected column holds the limb window of the constant convolution, scaled by 2^((cnv_offset + 1) * w) at any radix width w; the remaining limbs are zero-filled
    /// test       test_convolution_by_const, test_convolution_by_const_degree_rejected
    /// ```
    #[allow(clippy::too_many_arguments)]
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

    /// Returns the scratch bytes [`cnv_by_const_apply_add`](Convolution::cnv_by_const_apply_add) requires for those sizes.
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

    /// Adds the convolution of `a[a_col]` with coefficient `b_coeff` of `b[b_col]` to `res[res_col]`, leaving the limbs outside its support unchanged.
    ///
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
    #[allow(clippy::too_many_arguments)]
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
    /// Writes the convolution of `a[a_col]` and `b[b_col]` into `res[res_col]`, zero-filling the remaining limbs.
    ///
    /// ```text
    /// op         cnv_apply_dft(cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    /// class      basis
    /// mutation   out-of-place
    /// definition idft(res)[res_col,j] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = j + cnv_offset} a[a_col,u] * b[b_col,v]; other columns of res are unchanged
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of degree N; b: a CnvPVecR of degree N or of a degree dividing it; res.size(), a.size() and b.size() >= 1
    /// requires   scratch >= cnv_apply_dft_tmp_bytes(cnv_offset, res.size(), a.size(), b.size())
    /// ensures    the selected inverse column holds the limb window of the convolution, scaled by 2^((cnv_offset + 1) * w) at any radix width w; the remaining limbs are zero-filled
    /// sparse     b may be a prepared right operand of degree n, n a power of two dividing N and not below the backend's minimum sparse degree, prepared at degree n; res and a take degree N; the degree embedding of the api module docs defines the correspondence that reads it
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

    /// Returns the scratch bytes [`cnv_apply_dft_add`](Convolution::cnv_apply_dft_add) requires for those sizes.
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

    /// Adds the convolution of `a[a_col]` and `b[b_col]` to `res[res_col]`, leaving the limbs outside its support unchanged.
    ///
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
    #[allow(clippy::too_many_arguments)]
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

    /// Returns the scratch bytes [`cnv_apply_dft_sum`](Convolution::cnv_apply_dft_sum) requires for those sizes.
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

    /// Writes the sum of the convolutions of the `terms` into `res[res_col]`, zeroing the column when `terms` is empty.
    ///
    /// ```text
    /// op         cnv_apply_dft_sum(cnv_offset, res, res_col, terms, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition idft(res)[res_col,j] = sum_{0 <= t < terms.len()} sum_{0 <= u < terms[t].a.size(), 0 <= v < terms[t].b.size(), u + v = j + cnv_offset} terms[t].a[terms[t].a_col,u] * terms[t].b[terms[t].b_col,v]; an empty terms slice yields zero; other columns of res are unchanged
    /// domain     res: a VecZnxDft of degree N; terms: prepared left operands of degree N and right operands of degree N or of a degree dividing it, with their column indices
    /// requires   scratch >= cnv_apply_dft_sum_tmp_bytes(cnv_offset, res.size(), max({0} union {terms[t].a.size() : 0 <= t < terms.len()}), max({0} union {terms[t].b.size() : 0 <= t < terms.len()}))
    /// ensures    the selected inverse column is the sum of the selected convolution limb windows; an empty slice zeroes the column
    /// sparse     per term, as for cnv_apply_dft
    /// fallback   an empty slice zeroes the selected column; otherwise cnv_apply_dft overwrites with the first term and cnv_apply_dft_add accumulates the remaining terms
    /// override   allowed, with cnv_apply_dft_sum_tmp_bytes
    /// test       test_convolution_sum, test_cnv_apply_dft_sum_derived, test_convolution_sparse
    /// ```
    fn cnv_apply_dft_sum(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        terms: &[CnvDftAccTerm<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns the scratch bytes [`cnv_pairwise_apply_dft`](Convolution::cnv_pairwise_apply_dft) requires for those sizes.
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
    /// Writes the convolution of column `i` of `a` and `b` into `res[res_col]` when `i == j`, and the convolution of their two column sums otherwise.
    ///
    /// ```text
    /// op         cnv_pairwise_apply_dft(cnv_offset, res, res_col, a, b, i, j, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition for every 0 <= ell < res.size(), idft(res)[res_col,ell] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = ell + cnv_offset} a[i,u] * b[i,v] if i == j, and idft(res)[res_col,ell] = sum_{0 <= u < a.size(), 0 <= v < b.size(), u + v = ell + cnv_offset} (a[i,u] + a[j,u]) * (b[i,v] + b[j,v]) if i != j; other columns of res are unchanged
    /// domain     res: a VecZnxDft and a: a CnvPVecL, both of degree N; b: a CnvPVecR of degree N or of a degree dividing it; i, j column indices
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

    /// Returns the scratch bytes [`cnv_prepare_self`](Convolution::cnv_prepare_self) requires for those sizes.
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

    /// Writes each column of `a` into both `left` and `right`, truncated or zero-extended to their limb count.
    ///
    /// ```text
    /// op         cnv_prepare_self(left, right, a, scratch)
    /// class      derived
    /// mutation   out-of-place
    /// definition left[c,j] and right[c,j] read as a[c,j] for every 0 <= c < left.cols() and 0 <= j < left.size()
    /// domain     left: a CnvPVecL of degree N; right: a CnvPVecR of degree N, with right.cols() == left.cols() and right.size() == left.size(); a: a dense VecZnx of degree N with a.cols() == left.cols(), canonical at the precision the caller means to convolve at
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
