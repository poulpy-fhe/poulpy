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

/// Returns scratch bytes required for [`VmpApplyPMatSmallToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatSmallToDftTmpBytes {
    fn vmp_apply_pmat_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with a cold-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyPMatSmallToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatDftToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatDftToDftTmpBytes {
    fn vmp_apply_pmat_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with a cold-prepared matrix and `b` in DFT domain.
pub trait VmpApplyPMatDftToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatSmallToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatSmallToDftAccumulateTmpBytes {
    fn vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with a cold-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyPMatSmallToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatDftToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatDftToDftAccumulateTmpBytes {
    fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with a cold-prepared matrix and `b` in DFT domain.
pub trait VmpApplyPMatDftToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatSmallToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatSmallToBigTmpBytes {
    fn vmp_apply_pmat_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with a cold-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyPMatSmallToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatDftToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatDftToBigTmpBytes {
    fn vmp_apply_pmat_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with a cold-prepared matrix and `b` in DFT domain.
pub trait VmpApplyPMatDftToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatSmallToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatSmallToSmallTmpBytes {
    fn vmp_apply_pmat_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with a cold-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyPMatSmallToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyPMatDftToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyPMatDftToSmallTmpBytes {
    fn vmp_apply_pmat_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with a cold-prepared matrix and `b` in DFT domain.
pub trait VmpApplyPMatDftToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
