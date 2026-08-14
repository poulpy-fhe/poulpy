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

/// Zeroes all entries of a [`VmpTMat`](crate::layouts::VmpTMat).
pub trait VmpTMatZero<B: Backend> {
    fn vmp_tmat_zero(&self, res: &mut VmpTMatBackendMut<'_, B>);
}

/// Returns scratch bytes required for [`VmpApplyTMatSmallToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatSmallToDftTmpBytes {
    fn vmp_apply_tmat_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with a hot-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyTMatSmallToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatDftToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatDftToDftTmpBytes {
    fn vmp_apply_tmat_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with a hot-prepared matrix and `b` in DFT domain.
pub trait VmpApplyTMatDftToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatSmallToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatSmallToDftAccumulateTmpBytes {
    fn vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with a hot-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyTMatSmallToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatDftToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatDftToDftAccumulateTmpBytes {
    fn vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with a hot-prepared matrix and `b` in DFT domain.
pub trait VmpApplyTMatDftToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatSmallToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatSmallToBigTmpBytes {
    fn vmp_apply_tmat_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with a hot-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyTMatSmallToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatDftToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatDftToBigTmpBytes {
    fn vmp_apply_tmat_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with a hot-prepared matrix and `b` in DFT domain.
pub trait VmpApplyTMatDftToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatSmallToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatSmallToSmallTmpBytes {
    fn vmp_apply_tmat_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with a hot-prepared matrix and `b` in coefficient domain.
pub trait VmpApplyTMatSmallToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplyTMatDftToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplyTMatDftToSmallTmpBytes {
    fn vmp_apply_tmat_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with a hot-prepared matrix and `b` in DFT domain.
pub trait VmpApplyTMatDftToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
