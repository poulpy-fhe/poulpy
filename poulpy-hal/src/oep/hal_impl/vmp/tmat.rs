//! Open extension points whose prepared operand is the transformed hot-prep [`VmpTMat`](crate::layouts::VmpTMat).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScratchArena};

/// Backend extension points for the `tmat` tier of the VMP family.
///
/// # Safety
///
/// Implementations must uphold the backend safety contract; see
/// [`HalVmpImpl`](super::HalVmpImpl).
pub unsafe trait HalVmpTMatImpl<BE: Backend>: Backend {
    fn vmp_prepare_tmat_tmp_bytes(module: &Module<BE>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
    fn vmp_prepare_tmat(
        module: &Module<BE>,
        res: &mut crate::layouts::VmpTMatBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::VmpTMatBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    fn vmp_tmat_zero(module: &Module<BE>, res: &mut crate::layouts::VmpTMatBackendMut<'_, BE>);
}
