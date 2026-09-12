use crate::layouts::{
    Backend, PrepareHint, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned,
    VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
};

/// Allocates as [crate::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize, hint: PrepareHint) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize, hint: PrepareHint) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
pub trait SvpPrepare<B: Backend> {
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one prepared scalar polynomial column into another.
///
/// Copies representation bytes, so `res` and `a` must share the degree and the
/// [`PrepareHint`](crate::layouts::PrepareHint); every kernel asserts both.
pub trait SvpPPolCopy<B: Backend> {
    fn svp_ppol_copy(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize);
}

/// Returns scratch bytes required by [`SvpApplyDft::svp_apply_dft`].
pub trait SvpApplyDftTmpBytes {
    fn svp_apply_dft_tmp_bytes(&self, b_size: usize) -> usize;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
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
pub trait SvpApplyDftToDftAssign<B: Backend> {
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
