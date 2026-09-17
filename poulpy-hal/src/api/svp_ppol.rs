use crate::layouts::{
    Backend, PrepareHint, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned,
    VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
};

/// Allocates an [`SvpPPol`](crate::layouts::SvpPPol).
///
/// ```text
/// op         svp_ppol_alloc(cols, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1; hint: the PrepareHint the destination will be written under
/// ensures    returns an owned degree-N SvpPPol with cols columns and the requested hint; its contents are unspecified
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait SvpPPolAlloc<B: Backend> {
    /// Returns an owned [`SvpPPol`](crate::layouts::SvpPPol) with `cols` columns under `hint`.
    fn svp_ppol_alloc(&self, cols: usize, hint: PrepareHint) -> SvpPPolOwned<B>;
}

/// Returns the byte size of an [`SvpPPol`](crate::layouts::SvpPPol).
///
/// ```text
/// op         bytes_of_svp_ppol(cols, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1
/// ensures    returns the bytes required for a degree-N SvpPPol with cols columns and the requested hint
/// test       test_word_compat_prepare_hint_sizes, test_word_compat_svp_prepare_bytes
/// ```
pub trait SvpPPolBytesOf {
    /// Returns the bytes an [`SvpPPol`](crate::layouts::SvpPPol) with `cols` columns under `hint` occupies.
    fn bytes_of_svp_ppol(&self, cols: usize, hint: PrepareHint) -> usize;
}

/// Preparation of a [`ScalarZnx`](crate::layouts::ScalarZnx) into an [`SvpPPol`](crate::layouts::SvpPPol).
///
/// ```text
/// op         svp_prepare(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] reads as the polynomial a[a_col]; other columns of res are unchanged
/// domain     res: an SvpPPol; a: a ScalarZnx of the module degree
/// ensures    the selected scalar polynomial is prepared for multiplication
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpPrepare<B: Backend> {
    /// Writes `a[a_col]` into `res[res_col]` in the prepared representation.
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy of one prepared scalar polynomial column into another.
///
/// ```text
/// op         svp_ppol_copy(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] reads as the polynomial a[a_col]; other columns of res are unchanged
/// domain     res, a: SvpPPol of the same degree and the same PrepareHint
/// ensures    the selected prepared scalar polynomial is copied
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpPPolCopy<B: Backend> {
    /// Writes `a[a_col]` into `res[res_col]`.
    fn svp_ppol_copy(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize);
}

/// Returns scratch bytes required by [`SvpApplyDft::svp_apply_dft`].
///
/// ```text
/// op         svp_apply_dft_tmp_bytes(b_size)
/// class      support
/// mutation   none
/// domain     b_size: the coefficient-domain operand's limb count
/// ensures    returns the scratch bytes required by svp_apply_dft for this source limb count
/// test       test_svp_apply_dft
/// ```
pub trait SvpApplyDftTmpBytes {
    /// Returns the scratch bytes `svp_apply_dft` requires for a source of `b_size` limbs.
    fn svp_apply_dft_tmp_bytes(&self, b_size: usize) -> usize;
}

/// Product of a prepared scalar polynomial by a coefficient-domain vector, into the DFT domain.
///
/// ```text
/// op         svp_apply_dft(res, res_col, a, a_col, b, b_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = a[a_col] * b[b_col,j] in R_N; other columns of res are unchanged
/// domain     res: a VecZnxDft; a: an SvpPPol; b: a dense VecZnx; all operands have the module degree
/// requires   scratch >= svp_apply_dft_tmp_bytes(b.size())
/// ensures    the selected source limbs are multiplied by the prepared scalar polynomial; result limbs from b.size() onward are zero
/// fallback   transform b[b_col] into a one-column VecZnxDft with b.size() limbs, then multiply it by a[a_col]
/// override   allowed, with svp_apply_dft_tmp_bytes
/// test       test_svp_apply_dft, test_svp_apply_dft_derived
/// ```
pub trait SvpApplyDft<B: Backend> {
    /// Writes `a[a_col] * b[b_col]` into `res[res_col]`, zeroing the limbs from `b.size()` on.
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Product of a prepared scalar polynomial by a DFT-domain vector, in the DFT domain.
///
/// ```text
/// op         svp_apply_dft_to_dft(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = a[a_col] * idft(b)[b_col,j] in R_N; other columns of res are unchanged
/// domain     res, b: VecZnxDft; a: an SvpPPol; all operands have the module degree
/// ensures    the selected source limbs are multiplied by the prepared scalar polynomial; result limbs from b.size() onward are zero
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpApplyDftToDft<B: Backend> {
    /// Writes `a[a_col] * b[b_col]` into `res[res_col]`, zeroing the limbs from `b.size()` on.
    fn svp_apply_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    );
}

/// In-place product of a DFT-domain vector by a prepared scalar polynomial.
///
/// ```text
/// op         svp_apply_dft_to_dft_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition idft(res)[res_col,j] = a[a_col] * idft(old(res))[res_col,j] in R_N; other columns of res are unchanged
/// domain     res: a VecZnxDft; a: an SvpPPol; both operands have the module degree
/// ensures    every limb of the selected result column is multiplied by the prepared scalar polynomial
/// test       test_svp_apply_dft_to_dft_assign
/// ```
pub trait SvpApplyDftToDftAssign<B: Backend> {
    /// Multiplies every limb of `res[res_col]` by `a[a_col]`.
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
