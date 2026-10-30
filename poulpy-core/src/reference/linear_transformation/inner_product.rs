//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/guides/linear_transformation.md: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxDft`. Its bodies sit beside
//! [`DiagonalProd`](crate::layouts::DiagonalProd), whose provided product calls them; this module
//! holds the references of the batched backend hooks and the query.

use crate::layouts::BabyStepOperand;
use poulpy_hal::{
    api::Convolution,
    layouts::{Backend, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    LinearTransformationGiantStep,
    gadget::prod_dft_size,
    layouts::prepared::{PreparedDiagonal, glwe_accumulate_prepared_baby_steps_dft},
};

use super::LinearTransformationBabySteps;

/// Reference of [`LinearTransformationImpl::glwe_giant_step_batch`](crate::oep::LinearTransformationImpl::glwe_giant_step_batch),
/// which `impl_linear_transformation_reference_full!` forwards to: a giant step at a time.
pub fn glwe_giant_step_batch_reference<BE: Backend>(_module: &poulpy_hal::layouts::Module<BE>) -> usize {
    1
}

/// Reference of [`LinearTransformationImpl::glwe_accumulate_giant_steps`](crate::oep::LinearTransformationImpl::glwe_accumulate_giant_steps),
/// which `impl_linear_transformation_reference_full!` forwards to: each product accumulated on
/// its own with `cnv_apply_dft_sum`.
pub fn glwe_accumulate_giant_steps_reference<BE, M, L: BabyStepOperand<BE>>(
    module: &M,
    cnv_offset_hi: usize,
    prods: &mut [VecZnxDftBackendMut<'_, BE>],
    lhs: &LinearTransformationBabySteps<BE, L>,
    steps: &[LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>],
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: Convolution<BE>,
{
    assert_eq!(prods.len(), steps.len());
    for (prod, step) in prods.iter_mut().zip(steps) {
        glwe_accumulate_prepared_baby_steps_dft(module, cnv_offset_hi, prod, lhs, step, scratch);
    }
}

/// Scratch bytes required by the prepared PROD of one giant step, `glwe_accumulate_prepared_baby_steps_dft`.
pub(super) fn glwe_accumulate_prepared_baby_steps_dft_tmp_bytes<BE, M>(
    module: &M,
    cnv_offset_hi: usize,
    baby_size: usize,
    diagonal_size: usize,
) -> usize
where
    BE: Backend,
    M: Convolution<BE>,
{
    let res_dft_size = prod_dft_size(baby_size, diagonal_size, cnv_offset_hi);
    module.cnv_apply_dft_sum_tmp_bytes(cnv_offset_hi, res_dft_size, baby_size, diagonal_size)
}
