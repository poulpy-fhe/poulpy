//! SVP operations whose scalar operand is a packed cold-prep [`SvpPPol`](crate::layouts::SvpPPol).
use crate::layouts::{
    Backend, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned, VecZnxBackendMut,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
};

/// Allocates an [crate::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
pub trait SvpPreparePPol<B: Backend> {
    fn svp_prepare_ppol(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one packed prepared scalar polynomial column into another.
pub trait SvpPPolCopyBackend<B: Backend> {
    fn svp_ppol_copy_backend(
        &self,
        res: &mut SvpPPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}

svp_apply_trait!(
    /// `res = a * b`, with `a` cold-prepared and `b` in coefficient domain.
    SvpApplyPPolSmallToDft::svp_apply_ppol_small_to_dft(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);

svp_apply_trait!(
    /// `res = a * b`, with `a` cold-prepared and `b` in DFT domain.
    SvpApplyPPolDftToDft::svp_apply_ppol_dft_to_dft(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> dft
);

svp_apply_trait!(
    /// [`SvpApplyPPolSmallToDft`] followed by an inverse DFT.
    SvpApplyPPolSmallToBig::svp_apply_ppol_small_to_big(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);

svp_apply_trait!(
    /// [`SvpApplyPPolDftToDft`] followed by an inverse DFT.
    SvpApplyPPolDftToBig::svp_apply_ppol_dft_to_big(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> big
);

svp_apply_trait!(
    /// [`SvpApplyPPolSmallToBig`] followed by a normalization.
    SvpApplyPPolSmallToSmall::svp_apply_ppol_small_to_small(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);

svp_apply_trait!(
    /// [`SvpApplyPPolDftToBig`] followed by a normalization.
    SvpApplyPPolDftToSmall::svp_apply_ppol_dft_to_small(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> small
);

/// `res = a * res` with `a` cold-prepared.
pub trait SvpApplyPPolDftToDftAssign<B: Backend> {
    fn svp_apply_ppol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
