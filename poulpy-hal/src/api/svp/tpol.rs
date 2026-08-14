//! SVP operations whose scalar operand is a transformed hot-prep [`SvpTPol`](crate::layouts::SvpTPol).
use crate::layouts::{
    Backend, ScalarZnxBackendRef, ScratchArena, SvpTPolBackendMut, SvpTPolBackendRef, SvpTPolOwned, VecZnxBackendMut,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
};

/// Allocates an [crate::layouts::SvpTPol].
pub trait SvpTPolAlloc<B: Backend> {
    fn svp_tpol_alloc(&self, cols: usize) -> SvpTPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpTPol].
pub trait SvpTPolBytesOf {
    fn bytes_of_svp_tpol(&self, cols: usize) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpTPol].
pub trait SvpPrepareTPol<B: Backend> {
    fn svp_prepare_tpol(&self, res: &mut SvpTPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one transformed prepared scalar polynomial column into another.
pub trait SvpTPolCopyBackend<B: Backend> {
    fn svp_tpol_copy_backend(
        &self,
        res: &mut SvpTPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    );
}

svp_apply_trait!(
    /// `res = a * b`, with `a` hot-prepared and `b` in coefficient domain.
    SvpApplyTPolSmallToDft::svp_apply_tpol_small_to_dft(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);

svp_apply_trait!(
    /// `res = a * b`, with `a` hot-prepared and `b` in DFT domain.
    SvpApplyTPolDftToDft::svp_apply_tpol_dft_to_dft(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> dft
);

svp_apply_trait!(
    /// [`SvpApplyTPolSmallToDft`] followed by an inverse DFT.
    SvpApplyTPolSmallToBig::svp_apply_tpol_small_to_big(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);

svp_apply_trait!(
    /// [`SvpApplyTPolDftToDft`] followed by an inverse DFT.
    SvpApplyTPolDftToBig::svp_apply_tpol_dft_to_big(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> big
);

svp_apply_trait!(
    /// [`SvpApplyTPolSmallToBig`] followed by a normalization.
    SvpApplyTPolSmallToSmall::svp_apply_tpol_small_to_small(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);

svp_apply_trait!(
    /// [`SvpApplyTPolDftToBig`] followed by a normalization.
    SvpApplyTPolDftToSmall::svp_apply_tpol_dft_to_small(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> small
);

/// `res = a * res` with `a` hot-prepared.
pub trait SvpApplyTPolDftToDftAssign<B: Backend> {
    fn svp_apply_tpol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    );
}
