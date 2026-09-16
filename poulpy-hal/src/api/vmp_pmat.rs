use crate::layouts::{
    Backend, MatZnxBackendRef, PrepareHint, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
    VecZnxDftToBackendMut, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned,
};

/// Allocates a [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         vmp_pmat_alloc(rows, cols_in, cols_out, size, hint)
/// class      support
/// mutation   none
/// domain     every dimension >= 1; hint: the PrepareHint the destination will be written under
/// ensures    returns an owned degree-N VmpPMat of those dimensions in the backend's prepared representation, which is opaque; its contents are unspecified
/// exact      not an arithmetic operation
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait VmpPMatAlloc<B: Backend> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> VmpPMatOwned<B>;
}

/// Returns the byte size required for a [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         bytes_of_vmp_pmat(rows, cols_in, cols_out, size, hint)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the byte size of such a VmpPMat, the amount take_vmp_pmat_scratch carves. The hint never changes the value a prepared matrix denotes, and every backend gives it the same size
/// exact      not an arithmetic operation
/// test       test_word_compat_prepare_hint_sizes, test_word_compat_vmp_prepare_bytes
/// ```
pub trait VmpPMatBytesOf {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> usize;
}

/// Returns scratch bytes required for [`VmpPrepare`].
///
/// ```text
/// op         vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_prepare needs on a matrix of those dimensions
/// exact      not an arithmetic operation
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpPrepareTmpBytes {
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Prepares a coefficient-domain [`MatZnx`](crate::layouts::MatZnx) into a
/// DFT-domain [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         vmp_prepare(pmat, mat, scratch)
/// class      basis
/// mutation   out-of-place
/// domain     pmat: a VmpPMat; mat: a MatZnx of the same degree and dimensions
/// requires   scratch >= vmp_prepare_tmp_bytes(...)
/// ensures    pmat holds prep(mat) in the representation pmat's PrepareHint names. The representation is opaque, so the statement is on the observable: vmp_apply_dft_to_dft with it is the vector-matrix product by mat
/// exact      backend DFT class: exact for an NTT backend, approximate for a floating-point FFT backend
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpPrepare<B: Backend> {
    fn vmp_prepare(&self, pmat: &mut VmpPMatBackendMut<'_, B>, mat: &MatZnxBackendRef<'_, B>, scratch: &mut ScratchArena<'_, B>);
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDft`].
///
/// ```text
/// op         vmp_apply_dft_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_apply_dft needs: a min(a_size, b_rows)-limb, b_cols_in-column VecZnxDft for the transformed input, plus whatever vmp_apply_dft_to_dft asks for on the same shapes
/// exact      not an arithmetic operation
/// test       test_vmp_apply_dft
/// ```
pub trait VmpApplyDftTmpBytes {
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// Applies the vector-matrix product `VecZnx x VmpPMat -> VecZnxDft`.
///
/// ```text
/// op         vmp_apply_dft(res, a, pmat, scratch)
/// class      derived
/// mutation   out-of-place
/// definition vmp_apply_dft_to_dft(res, vec_znx_dft_apply(1, 0, a), pmat, limb_offset = 0)
/// domain     res: a VecZnxDft of pmat.cols_out() columns; a: a dense VecZnx of the module degree; pmat: a VmpPMat
/// requires   scratch >= vmp_apply_dft_tmp_bytes(...)
/// ensures    idft(res) = [[a]] * M, the matrix pmat was prepared from; the min(a.size(), pmat.rows()) leading limbs of a are consumed and a's trailing columns are aligned with pmat.cols_in(), the leading ones zeroed
/// fallback   OEP default body: zero the unaligned leading columns, transform the consumed limbs into a carved VecZnxDft, then apply in the DFT domain
/// override   allowed, with vmp_apply_dft_tmp_bytes
/// exact      backend DFT class: exact for an NTT backend, approximate for a floating-point FFT backend
/// test       test_vmp_apply_dft, test_vmp_apply_dft_derived
/// ```
pub trait VmpApplyDft<B: Backend> {
    fn vmp_apply_dft<R>(
        &self,
        res: &mut R,
        a: &VecZnxBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) where
        R: VecZnxDftToBackendMut<B>;
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDftToDft`].
///
/// ```text
/// op         vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_apply_dft_to_dft needs on those shapes
/// exact      not an arithmetic operation
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpApplyDftToDftTmpBytes {
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDftToDftAdd`].
///
/// ```text
/// op         vmp_apply_dft_to_dft_add_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_apply_dft_to_dft_add needs: a res_size-limb, b_cols_out-column staging accumulator plus the product's own scratch. A backend that overrides the operation with a fused kernel overrides this too, and may report less
/// exact      not an arithmetic operation
/// test       test_vmp_apply_dft_to_dft_add
/// ```
pub trait VmpApplyDftToDftAddTmpBytes {
    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// ```text
/// op         vmp_apply_dft_to_dft(res, a, pmat, limb_offset, scratch)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VecZnxDft of the module degree; pmat: a VmpPMat; where a dimension disagrees the largest valid one is used
/// requires   scratch >= vmp_apply_dft_to_dft_tmp_bytes(...)
/// ensures    idft(res) = idft(a) * M, the matrix pmat was prepared from, reading pmat's limbs from limb_offset on; row i of the product weighs limb i of a. Only the limbs the product reaches are written
/// exact      backend DFT class: exact for an NTT backend, approximate for a floating-point FFT backend
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpApplyDftToDft<B: Backend> {
    /// Applies the vector matrix product [crate::layouts::VecZnxDft] x [crate::layouts::VmpPMat].
    ///
    /// A vector matrix product numerically equivalent to a sum of [crate::api::SvpApplyDft],
    /// where each [crate::layouts::SvpPPol] is a limb of the input [crate::layouts::VecZnx] in DFT,
    /// and each vector a [crate::layouts::VecZnxDft] (row) of the [crate::layouts::VmpPMat].
    ///
    /// As such, given an input [crate::layouts::VecZnx] of `i` size and a [crate::layouts::VmpPMat] of `i` rows and
    /// `j` size, the output is a [crate::layouts::VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [crate::layouts::VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [crate::layouts::VecZnxDft].
    /// * `a`: the left operand [crate::layouts::VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [crate::layouts::VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpApplyDftToDftTmpBytes::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft<'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vmp_apply_dft_to_dft_add(res, a, pmat, limb_offset, scratch)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_dft_add_assign(res, vmp_apply_dft_to_dft(tmp, a, pmat, limb_offset)) over every output column, tmp of res.size() limbs
/// domain     res, a: VecZnxDft of the module degree; pmat: a VmpPMat
/// requires   scratch >= vmp_apply_dft_to_dft_add_tmp_bytes(...)
/// ensures    res gains idft(a) * M over the same limb window vmp_apply_dft_to_dft writes; limbs the product does not reach gain zero and so keep their value
/// fallback   OEP default body: a zeroed res.size()-limb staging accumulator, the product into it, then a column-wise dft_add_assign. The zeroing is load-bearing: the product may leave the limbs past its bound untouched, and an unzeroed accumulator would fold scratch into res there
/// override   allowed, with vmp_apply_dft_to_dft_add_tmp_bytes
/// exact      backend DFT class: exact for an NTT backend, approximate for a floating-point FFT backend
/// test       test_vmp_apply_dft_to_dft_add, test_vmp_apply_dft_to_dft_add_derived
/// ```
pub trait VmpApplyDftToDftAdd<B: Backend> {
    /// Fused `res += a · pmat`, shifted by `limb_offset` limbs.
    fn vmp_apply_dft_to_dft_add<'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Copies selected rows and the leading limbs of a
/// [`VmpPMat`](crate::layouts::VmpPMat) into a smaller one.
///
/// Row `i` of `res` is row `first_row + i * row_step` of `a`, truncated to
/// `res.size()` limbs. Only the selected rows and limbs are read, so the result
/// is a dense prepared matrix over exactly the material a coarsened gadget
/// decomposition uses.
///
/// Every kernel validates the selection first via `assert_extractable`:
/// matching [`PrepareHint`](crate::layouts::PrepareHint) (the copy moves
/// representation bytes), matching `n` and both column counts, `res.size() <= a.size()`,
/// `row_step > 0`, and a last row that is inside `a` without overflowing. Past
/// that check the kernel may index on those facts without bounds checks.
///
/// ```text
/// op         vmp_extract_selected_rows(res, a, first_row, row_step)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VmpPMat of the same degree, the same column counts and the same PrepareHint, with res.size() <= a.size(), row_step > 0 and a last selected row inside a; assert_extractable checks all of it before the kernel indexes on those facts
/// ensures    row i of res is row first_row + i * row_step of a, truncated to res.size() limbs, so res is a dense prepared matrix over exactly the material a coarsened gadget decomposition uses. It moves representation bytes
/// exact      exact, it copies
/// test       test_vmp_extract_selected_rows
/// ```
pub trait VmpExtractSelectedRows<B: Backend> {
    fn vmp_extract_selected_rows(
        &self,
        res: &mut VmpPMatBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        first_row: usize,
        row_step: usize,
    );
}

/// Zeroes all entries of a [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         vmp_zero(res)
/// class      basis
/// mutation   out-of-place
/// domain     res: a VmpPMat
/// ensures    every entry of res is the representation of zero, so a vector-matrix product through it yields zero
/// exact      exact
/// test       test_vmp_zero
/// ```
pub trait VmpZero<B: Backend> {
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>);
}
