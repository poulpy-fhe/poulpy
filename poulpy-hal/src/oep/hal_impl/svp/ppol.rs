//! Open extension points whose prepared operand is the packed cold-prep [`SvpPPol`](crate::layouts::SvpPPol).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScalarZnxBackendRef};

/// Backend extension points for the `ppol` tier of the SVP family.
///
/// # Safety
///
/// Implementations must uphold the backend safety contract; see
/// [`HalSvpImpl`](super::HalSvpImpl).
pub unsafe trait HalSvpPPolImpl<BE: Backend>: Backend {
    fn svp_prepare_ppol(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );
    fn svp_ppol_copy_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    fn svp_apply_ppol_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );
}
