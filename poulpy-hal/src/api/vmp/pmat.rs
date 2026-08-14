//! VMP operations whose matrix operand is a packed cold-prep [`VmpPMat`](crate::layouts::VmpPMat).
use crate::layouts::{
    Backend, MatZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned,
};

/// Allocates a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPMatAlloc<B: Backend> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>;
}

/// Returns the byte size required for a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPMatBytesOf {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Returns scratch bytes required for [`VmpPreparePMat`].
pub trait VmpPreparePMatTmpBytes {
    fn vmp_prepare_pmat_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Prepares a coefficient-domain [`MatZnx`](crate::layouts::MatZnx) into a
/// packed cold-prep [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPreparePMat<B: Backend> {
    fn vmp_prepare_pmat(
        &self,
        res: &mut VmpPMatBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

vmp_apply_trait!(
    /// `res = a * b`, with a cold-prepared matrix and `b` in coefficient domain.
    VmpApplyPMatSmallToDft::vmp_apply_pmat_small_to_dft(VmpPMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, with a cold-prepared matrix and `b` in DFT domain.
    VmpApplyPMatDftToDft::vmp_apply_pmat_dft_to_dft(VmpPMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with a cold-prepared matrix and `b` in coefficient domain.
    VmpApplyPMatSmallToDftAccumulate::vmp_apply_pmat_small_to_dft_accumulate(
        VmpPMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>
    ) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with a cold-prepared matrix and `b` in DFT domain.
    VmpApplyPMatDftToDftAccumulate::vmp_apply_pmat_dft_to_dft_accumulate(
        VmpPMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with a cold-prepared matrix and `b` in coefficient domain.
    VmpApplyPMatSmallToBig::vmp_apply_pmat_small_to_big(VmpPMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with a cold-prepared matrix and `b` in DFT domain.
    VmpApplyPMatDftToBig::vmp_apply_pmat_dft_to_big(VmpPMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with a cold-prepared matrix and `b` in coefficient domain.
    VmpApplyPMatSmallToSmall::vmp_apply_pmat_small_to_small(VmpPMatBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with a cold-prepared matrix and `b` in DFT domain.
    VmpApplyPMatDftToSmall::vmp_apply_pmat_dft_to_small(
        VmpPMatBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> small
);
