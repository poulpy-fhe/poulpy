use crate::layouts::{
    Backend, PrepareHint, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned,
    VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
};

/// Allocates as [crate::layouts::SvpPPol].
///
/// ```text
/// op         svp_ppol_alloc(cols, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1; hint: the PrepareHint the destination will be written under
/// ensures    returns an owned degree-N SvpPPol of `cols` columns in the backend's prepared representation, which is opaque; its contents are unspecified
/// test       test_word_compat_prepare_hint_sizes
/// ```
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize, hint: PrepareHint) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
///
/// ```text
/// op         bytes_of_svp_ppol(cols, hint)
/// class      support
/// mutation   none
/// domain     cols >= 1
/// ensures    returns the byte size of such an SvpPPol, the amount take_svp_ppol_scratch carves. The hint never changes the value a prepared operand denotes, and every backend gives it the same size
/// test       test_word_compat_prepare_hint_sizes, test_word_compat_svp_prepare_bytes
/// ```
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize, hint: PrepareHint) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
///
/// ```text
/// op         svp_prepare(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] = a[a_col][0]
/// domain     res: an SvpPPol; a: a ScalarZnx of the module degree
/// ensures    res[res_col] holds prep(a[a_col]) in the representation res's PrepareHint names. The representation is opaque, so the statement is on the observable: svp_apply_dft_to_dft with it multiplies by a[a_col] in the ring
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpPrepare<B: Backend> {
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one prepared scalar polynomial column into another.
///
/// Copies representation bytes, so `res` and `a` must share the degree and the
/// [`PrepareHint`](crate::layouts::PrepareHint); every kernel asserts both.
///
/// ```text
/// op         svp_ppol_copy(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] = a[a_col]
/// domain     res, a: SvpPPol of the same degree and the same PrepareHint, both asserted by the kernel, since the copy moves representation bytes
/// ensures    res[res_col] denotes what a[a_col] denotes
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpPPolCopy<B: Backend> {
    fn svp_ppol_copy(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize);
}

/// Returns scratch bytes required by [`SvpApplyDft::svp_apply_dft`].
///
/// ```text
/// op         svp_apply_dft_tmp_bytes(b_size)
/// class      support
/// mutation   none
/// domain     b_size: the coefficient-domain operand's limb count
/// ensures    returns the scratch bytes svp_apply_dft needs: one b_size-limb, one-column VecZnxDft for the transformed right operand
/// test       test_svp_apply_dft
/// ```
pub trait SvpApplyDftTmpBytes {
    fn svp_apply_dft_tmp_bytes(&self, b_size: usize) -> usize;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
///
/// ```text
/// op         svp_apply_dft(res, res_col, a, a_col, b, b_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition svp_apply_dft_to_dft(res, res_col, a, a_col, vec_znx_dft_apply(1, 0, b, b_col), 0)
/// domain     res: a VecZnxDft; a: an SvpPPol; b: a dense VecZnx of the module degree
/// requires   scratch >= svp_apply_dft_tmp_bytes(b.size())
/// ensures    idft(res[res_col]) = a[a_col] * b[b_col] in the ring, over min(res.size(), b.size()) limbs; the limbs of res past that are zero
/// fallback   OEP default body: transform b into a carved VecZnxDft, then apply in the DFT domain
/// override   allowed, with svp_apply_dft_tmp_bytes
/// test       test_svp_apply_dft, test_svp_apply_dft_derived
/// ```
pub trait SvpApplyDft<B: Backend> {
    /// `res[res_col] = a[a_col] * dft(b[b_col])`. Limbs of `res` beyond
    /// `min(res.size(), b.size())` are zeroed.
    ///
    /// `scratch` must hold at least
    /// [`svp_apply_dft_tmp_bytes`](SvpApplyDftTmpBytes::svp_apply_dft_tmp_bytes)
    /// on `b.size()`: the derived body carves one `b.size()`-limb, one-column
    /// `VecZnxDft` for the transformed right operand.
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

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
///
/// ```text
/// op         svp_apply_dft_to_dft(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res[res_col])[j] = a[a_col] * idft(b[b_col])[j]
/// domain     res, b: VecZnxDft of the module degree; a: an SvpPPol
/// ensures    idft(res[res_col]) = a[a_col] * idft(b[b_col]) in the ring, limb by limb; limbs of res past b.size() are zero
/// test       test_svp_apply_dft_to_dft
/// ```
pub trait SvpApplyDftToDft<B: Backend> {
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

/// Apply a scalar-vector product between `res[res_col]` and `a[a_col]` and stores the result on `res[res_col]`.
///
/// ```text
/// op         svp_apply_dft_to_dft_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition svp_apply_dft_to_dft(res, res_col, a, a_col, res, res_col)
/// domain     res: a VecZnxDft of the module degree; a: an SvpPPol
/// ensures    idft(res[res_col]) is multiplied by a[a_col] in the ring, limb by limb
/// test       test_svp_apply_dft_to_dft_assign
/// ```
pub trait SvpApplyDftToDftAssign<B: Backend> {
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
