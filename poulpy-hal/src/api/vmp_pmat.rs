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
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait VmpPMatAlloc<B: Backend> {
    /// Returns an owned [`VmpPMat`](crate::layouts::VmpPMat) of the given dimensions under `hint`.
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> VmpPMatOwned<B>;
}

/// Returns the byte size of a [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         bytes_of_vmp_pmat(rows, cols_in, cols_out, size, hint)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the byte size required for the given prepared matrix dimensions and hint
/// test       test_word_compat_prepare_hint_sizes, test_word_compat_vmp_prepare_bytes
/// ```
pub trait VmpPMatBytesOf {
    /// Returns the bytes a [`VmpPMat`](crate::layouts::VmpPMat) of the given dimensions under `hint` occupies.
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> usize;
}

/// Returns the scratch bytes [`VmpPrepare`] requires.
///
/// ```text
/// op         vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_prepare needs on a matrix of those dimensions
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpPrepareTmpBytes {
    /// Returns the scratch bytes `vmp_prepare` requires for a matrix of the given dimensions.
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Preparation of a coefficient-domain [`MatZnx`](crate::layouts::MatZnx) into a [`VmpPMat`](crate::layouts::VmpPMat).
///
/// ```text
/// op         vmp_prepare(pmat, mat, scratch)
/// class      basis
/// mutation   out-of-place
/// definition pmat[r,c,d,j] reads as mat[r,c,d,j] for every 0 <= r < pmat.rows(), 0 <= c < pmat.cols_in(), 0 <= d < pmat.cols_out() and 0 <= j < pmat.size()
/// domain     pmat: a VmpPMat; mat: a MatZnx of the same degree and dimensions
/// requires   scratch >= vmp_prepare_tmp_bytes(pmat.rows(), pmat.cols_in(), pmat.cols_out(), pmat.size())
/// ensures    every prepared matrix entry reads the corresponding source entry
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpPrepare<B: Backend> {
    /// Writes every entry of `mat` into `pmat` in the prepared representation.
    fn vmp_prepare(&self, pmat: &mut VmpPMatBackendMut<'_, B>, mat: &MatZnxBackendRef<'_, B>, scratch: &mut ScratchArena<'_, B>);
}

#[allow(clippy::too_many_arguments)]
/// Returns the scratch bytes [`VmpApplyDft`] requires.
///
/// ```text
/// op         vmp_apply_dft_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_apply_dft needs on those shapes
/// test       test_vmp_apply_dft
/// ```
pub trait VmpApplyDftTmpBytes {
    /// Returns the scratch bytes `vmp_apply_dft` requires for the given shapes.
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

/// Vector-matrix product of a coefficient-domain vector by a prepared matrix, into the DFT domain.
///
/// ```text
/// op         vmp_apply_dft(res, a, pmat, scratch)
/// class      derived
/// mutation   out-of-place
/// definition idft(res)[d,j] = sum_{0 <= i < min(a.size(), pmat.rows()), max(pmat.cols_in() - a.cols(), 0) <= c < pmat.cols_in()} a[c + a.cols() - pmat.cols_in(),i] * pmat[i,c,d,j] for every 0 <= d < res.cols()
/// domain     res: a VecZnxDft of pmat.cols_out() columns; a: a dense VecZnx; pmat: a VmpPMat; all operands have the degree N of the call
/// requires   scratch >= vmp_apply_dft_tmp_bytes(res.size(), a.size(), pmat.rows(), pmat.cols_in(), pmat.cols_out(), pmat.size())
/// ensures    leading input limbs pair with matrix rows, trailing input columns align with the highest matrix input columns, and output limbs past pmat.size() are zero
/// fallback   zero unaligned leading columns, transform the consumed limbs into a temporary VecZnxDft, then apply in the DFT domain
/// override   allowed, with vmp_apply_dft_tmp_bytes
/// test       test_vmp_apply_dft, test_vmp_apply_dft_derived
/// ```
pub trait VmpApplyDft<B: Backend> {
    /// Writes `a * pmat` into `res`.
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
/// Returns the scratch bytes [`VmpApplyDftToDft`] requires.
///
/// ```text
/// op         vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes vmp_apply_dft_to_dft needs on those shapes
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpApplyDftToDftTmpBytes {
    /// Returns the scratch bytes `vmp_apply_dft_to_dft` requires for the given shapes.
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
/// Returns the scratch bytes [`VmpApplyDftToDftAdd`] requires.
///
/// ```text
/// op         vmp_apply_dft_to_dft_add_tmp_bytes(res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns the scratch bytes required by vmp_apply_dft_to_dft_add for the given shapes
/// test       test_vmp_apply_dft_to_dft_add
/// ```
pub trait VmpApplyDftToDftAddTmpBytes {
    /// Returns the scratch bytes `vmp_apply_dft_to_dft_add` requires for the given shapes.
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

/// Vector-matrix product of a DFT-domain vector by a prepared matrix, in the DFT domain.
///
/// ```text
/// op         vmp_apply_dft_to_dft(res, a, pmat, limb_offset, scratch)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[d,j] = sum_{0 <= i < min(a.size(), pmat.rows()), 0 <= c < pmat.cols_in()} idft(a)[c,i] * pmat[i,c,d,j + limb_offset] for every 0 <= d < res.cols()
/// domain     res, a: VecZnxDft of degree N; pmat: a VmpPMat of the same degree; a.cols() == pmat.cols_in(), res.cols() == pmat.cols_out()
/// requires   scratch >= vmp_apply_dft_to_dft_tmp_bytes(res.size(), a.size(), pmat.rows(), pmat.cols_in(), pmat.cols_out(), pmat.size())
/// ensures    limb i of a pairs with matrix row i; result limb j reads matrix limb j + limb_offset; limbs j >= max(pmat.size() - limb_offset, 0) are zero
/// test       test_vmp_apply_dft_to_dft
/// ```
pub trait VmpApplyDftToDft<B: Backend> {
    /// Writes `a * pmat` into `res`, reading the matrix limbs from `limb_offset` on.
    fn vmp_apply_dft_to_dft<'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Accumulating vector-matrix product of a DFT-domain vector by a prepared matrix, in the DFT domain.
///
/// ```text
/// op         vmp_apply_dft_to_dft_add(res, a, pmat, limb_offset, scratch)
/// class      derived
/// mutation   accumulate
/// definition idft(res)[d,j] = idft(old(res))[d,j] + sum_{0 <= i < min(a.size(), pmat.rows()), 0 <= c < pmat.cols_in()} idft(a)[c,i] * pmat[i,c,d,j + limb_offset] for every 0 <= d < res.cols(); limbs j >= max(pmat.size() - limb_offset, 0) are unchanged, and all of res is unchanged when the sum index set is empty
/// domain     res, a: VecZnxDft of degree N; pmat: a VmpPMat of the same degree; a.cols() == pmat.cols_in(), res.cols() == pmat.cols_out()
/// requires   scratch >= vmp_apply_dft_to_dft_add_tmp_bytes(res.size(), a.size(), pmat.rows(), pmat.cols_in(), pmat.cols_out(), pmat.size())
/// ensures    every output column gains the matrix product; limbs j >= max(pmat.size() - limb_offset, 0) retain their old value
/// fallback   zero a res.cols()-column res.size()-limb temporary, apply the product into it, then add each temporary column to its corresponding destination column
/// override   allowed, with vmp_apply_dft_to_dft_add_tmp_bytes
/// test       test_vmp_apply_dft_to_dft_add, test_vmp_apply_dft_to_dft_add_derived
/// ```
pub trait VmpApplyDftToDftAdd<B: Backend> {
    /// Adds `a * pmat` to `res`, reading the matrix limbs from `limb_offset` on.
    fn vmp_apply_dft_to_dft_add<'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Copy of selected rows and the leading limbs of a [`VmpPMat`](crate::layouts::VmpPMat) into a smaller one.
///
/// ```text
/// op         vmp_extract_selected_rows(res, a, first_row, row_step)
/// class      basis
/// mutation   out-of-place
/// definition res[r,c,d,j] reads as a[first_row + r * row_step,c,d,j] for every 0 <= r < res.rows(), 0 <= c < res.cols_in() and 0 <= d < res.cols_out()
/// domain     res, a: VmpPMat of the same degree, column counts and PrepareHint; res.size() <= a.size(), row_step > 0; res.rows() == 0 or first_row + (res.rows() - 1) * row_step < a.rows() without index overflow
/// ensures    the selected rows and leading res.size() limbs retain their source values
/// test       test_vmp_extract_selected_rows
/// ```
pub trait VmpExtractSelectedRows<B: Backend> {
    /// Writes row `first_row + r * row_step` of `a` into row `r` of `res`.
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
/// definition res[r,c,d,j] reads as 0 for every 0 <= r < res.rows(), 0 <= c < res.cols_in() and 0 <= d < res.cols_out()
/// domain     res: a VmpPMat
/// ensures    every prepared matrix entry reads as zero
/// test       test_vmp_zero
/// ```
pub trait VmpZero<B: Backend> {
    /// Writes zero into every entry of `res`.
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>);
}
