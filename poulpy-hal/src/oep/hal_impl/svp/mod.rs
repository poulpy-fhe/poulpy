//! Open extension points for the SVP family.

#![allow(clippy::too_many_arguments)]

mod ppol;
mod tpol;

pub use ppol::HalSvpPPolImpl;
pub use tpol::HalSvpTPolImpl;

use crate::layouts::{Backend, Module};

/// Scalar-vector product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared
/// polynomial layouts and arithmetic correctness.
pub unsafe trait HalSvpImpl<BE: Backend>: HalSvpPPolImpl<BE> + HalSvpTPolImpl<BE> {
    fn svp_apply_to_big_tmp_bytes(module: &Module<BE>, res_size: usize) -> usize;

    fn svp_apply_to_small_tmp_bytes(module: &Module<BE>, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    fn svp_apply_small_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );
}
