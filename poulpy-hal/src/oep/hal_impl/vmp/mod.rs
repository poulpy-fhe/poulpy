//! Open extension points for the VMP family.

#![allow(clippy::too_many_arguments)]

mod pmat;
mod tmat;

pub use pmat::HalVmpPMatImpl;
pub use tmat::HalVmpTMatImpl;

use crate::layouts::{Backend, Module};

/// Vector-matrix product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared matrix
/// layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalVmpImpl<BE: Backend>: HalVmpPMatImpl<BE> + HalVmpTMatImpl<BE> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
}
