//! VMP operations whose matrix operand is a transformed hot-prep [`VmpTMat`](crate::layouts::VmpTMat).
use crate::layouts::{
    Backend, MatZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VmpTMatBackendMut, VmpTMatBackendRef, VmpTMatOwned,
};

/// Allocates a [`VmpTMat`](crate::layouts::VmpTMat).
pub trait VmpTMatAlloc<B: Backend> {
    fn vmp_tmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpTMatOwned<B>;
}

/// Returns the byte size required for a [`VmpTMat`](crate::layouts::VmpTMat).
pub trait VmpTMatBytesOf {
    fn bytes_of_vmp_tmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Returns scratch bytes required for [`VmpPrepareTMat`].
pub trait VmpPrepareTMatTmpBytes {
    fn vmp_prepare_tmat_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Prepares a coefficient-domain [`MatZnx`](crate::layouts::MatZnx) into a
/// transformed hot-prep [`VmpTMat`](crate::layouts::VmpTMat).
pub trait VmpPrepareTMat<B: Backend> {
    fn vmp_prepare_tmat(
        &self,
        res: &mut VmpTMatBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

vmp_apply_trait!(
    /// `res = a * b`, with a hot-prepared matrix and `b` in coefficient domain.
    VmpApplyTMatSmallToDft::vmp_apply_tmat_small_to_dft(VmpTMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, with a hot-prepared matrix and `b` in DFT domain.
    VmpApplyTMatDftToDft::vmp_apply_tmat_dft_to_dft(VmpTMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with a hot-prepared matrix and `b` in coefficient domain.
    VmpApplyTMatSmallToDftAccumulate::vmp_apply_tmat_small_to_dft_accumulate(
        VmpTMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>
    ) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with a hot-prepared matrix and `b` in DFT domain.
    VmpApplyTMatDftToDftAccumulate::vmp_apply_tmat_dft_to_dft_accumulate(
        VmpTMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with a hot-prepared matrix and `b` in coefficient domain.
    VmpApplyTMatSmallToBig::vmp_apply_tmat_small_to_big(VmpTMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with a hot-prepared matrix and `b` in DFT domain.
    VmpApplyTMatDftToBig::vmp_apply_tmat_dft_to_big(VmpTMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with a hot-prepared matrix and `b` in coefficient domain.
    VmpApplyTMatSmallToSmall::vmp_apply_tmat_small_to_small(VmpTMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with a hot-prepared matrix and `b` in DFT domain.
    VmpApplyTMatDftToSmall::vmp_apply_tmat_dft_to_small(
        VmpTMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> small
);
