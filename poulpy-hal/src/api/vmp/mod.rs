//! Vector-matrix product over the operand domains.
//!
//! A method name lists the domain of each operand, then the output domain:
//!
//! ```text
//! vmp_apply_<matrix-domain>_<vector-domain>_to_<output-domain>
//! ```
//!
//! Each domain token is the layout actually passed. The matrix operand:
//!
//! - `small`: coefficient domain, a [`MatZnx`](crate::layouts::MatZnx).
//! - `tmat`: transformed hot-prep, a [`VmpTMat`](crate::layouts::VmpTMat).
//! - `pmat`: packed cold-prep, a [`VmpPMat`](crate::layouts::VmpPMat).
//!
//! The vector operand: `small` for [`VecZnx`](crate::layouts::VecZnx), `dft`
//! for [`VecZnxDft`](crate::layouts::VecZnxDft). The output: `dft`,
//! `dft_accumulate` (adding into `res` instead of overwriting), `big` (IDFT of
//! the `dft` result) or `small` (normalization of the `big` result).
//!
//! The matrix leads in both the name and the signature. `limb_offset` is
//! carried only by the variants whose vector operand is already in DFT domain.
//!
//! The `small` matrix variants prepare the whole matrix on every call, so they
//! are one-shot paths: prepare into a `VmpTMat` or `VmpPMat` when the same
//! matrix is applied more than once.

use crate::layouts::{
    Backend, MatZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VmpPMatBackendMut,
};

/// Zeroes all entries of a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpZero<B: Backend> {
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>);
}

/// Returns scratch bytes required for [`VmpApplySmallSmallToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallSmallToDftTmpBytes {
    fn vmp_apply_small_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with an unprepared matrix and `b` in coefficient domain.
pub trait VmpApplySmallSmallToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallDftToDft`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallDftToDftTmpBytes {
    fn vmp_apply_small_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, with an unprepared matrix and `b` in DFT domain.
pub trait VmpApplySmallDftToDft<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallSmallToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallSmallToDftAccumulateTmpBytes {
    fn vmp_apply_small_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with an unprepared matrix and `b` in coefficient domain.
pub trait VmpApplySmallSmallToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallDftToDftAccumulate`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallDftToDftAccumulateTmpBytes {
    fn vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res += a * b`, with an unprepared matrix and `b` in DFT domain.
pub trait VmpApplySmallDftToDftAccumulate<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallSmallToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallSmallToBigTmpBytes {
    fn vmp_apply_small_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with an unprepared matrix and `b` in coefficient domain.
pub trait VmpApplySmallSmallToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallDftToBig`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallDftToBigTmpBytes {
    fn vmp_apply_small_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT applied, with an unprepared matrix and `b` in DFT domain.
pub trait VmpApplySmallDftToBig<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallSmallToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallSmallToSmallTmpBytes {
    fn vmp_apply_small_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with an unprepared matrix and `b` in coefficient domain.
pub trait VmpApplySmallSmallToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for [`VmpApplySmallDftToSmall`].
#[allow(clippy::too_many_arguments)]
pub trait VmpApplySmallDftToSmallTmpBytes {
    fn vmp_apply_small_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
}

/// `res = a * b`, IDFT and normalization applied, with an unprepared matrix and `b` in DFT domain.
pub trait VmpApplySmallDftToSmall<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

mod pmat;
mod tmat;

pub use pmat::*;
pub use tmat::*;
