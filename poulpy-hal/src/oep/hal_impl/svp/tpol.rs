//! Open extension points whose prepared operand is the transformed hot-prep [`SvpTPol`](crate::layouts::SvpTPol).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScalarZnxBackendRef};

/// Backend extension points for the `tpol` tier of the SVP family.
///
/// # Safety
///
/// Implementations must uphold the backend safety contract; see
/// [`HalSvpImpl`](super::HalSvpImpl).
pub unsafe trait HalSvpTPolImpl<BE: Backend>: Backend {
    fn svp_prepare_tpol(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );
    fn svp_tpol_copy_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
    fn svp_apply_tpol_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    );
}
