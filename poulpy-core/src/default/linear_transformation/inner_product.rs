//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/lt_bsgs.md §6.3: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxDft`.
//! `cnv_apply_dft` overwrites its destination, so the first term initializes
//! each DFT column and following terms are accumulated explicitly.

use poulpy_hal::{
    api::{Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftCopy},
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, ScratchArena, VecZnxDftBackendMut, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, ZnxInfos,
    },
};

use super::{GLWEPreparedLinearTransform, GLWEPreparedLinearTransformGiantStep, baby_steps::GLWEPreparedBabyStepHelper};

/// PROD block for one giant step, kept in DFT domain.
///
/// Leaves the `(r+1)` output columns in `prod_dft`. Convolution output is staged
/// through a one-column DFT scratch so all backends follow the same
/// column-selection path; the first term is copied into `prod_dft`, and later
/// terms are accumulated with `vec_znx_dft_add_assign`.
pub(super) fn glwe_accumulate_prepared_baby_steps_dft<BE, M, B>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prepared: &GLWEPreparedLinearTransform<BE>,
    gs: &GLWEPreparedLinearTransformGiantStep<BE>,
    babies: &B,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN + Convolution<BE> + VecZnxDftAddAssign<BE> + VecZnxDftBytesOf + VecZnxDftCopy<BE>,
    B: GLWEPreparedBabyStepHelper<BE>,
{
    let first_baby_rot = prepared.baby_step_rotation(gs.first_baby_step_index());
    let sizing_diagonal_operand = gs.diagonal(first_baby_rot);
    let first_baby_operand = babies.baby_step(first_baby_rot);
    let cols = first_baby_operand.cols();
    let res_dft_size = first_baby_operand.size() + sizing_diagonal_operand.size() - cnv_offset_hi;
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    let (mut term_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_dft_size);

    for col in 0..cols {
        for (term_idx, &baby_step_idx) in gs.baby_step_indexes().iter().enumerate() {
            let baby_rot = prepared.baby_step_rotation(baby_step_idx);
            let diagonal = gs.diagonal(baby_rot);
            let baby = babies.baby_step(baby_rot);
            assert_eq!(baby.cols(), cols);
            assert_eq!(baby.size() + diagonal.size() - cnv_offset_hi, res_dft_size);

            module.cnv_apply_dft(
                cnv_offset_hi,
                &mut term_dft.to_backend_mut(),
                0,
                &baby.to_backend_ref(),
                col,
                &diagonal.to_backend_ref(),
                0,
                &mut scratch_1.borrow(),
            );

            if term_idx == 0 {
                module.vec_znx_dft_copy(1, 0, prod_dft, col, &term_dft.to_backend_ref(), 0);
            } else {
                module.vec_znx_dft_add_assign(prod_dft, col, &term_dft.to_backend_ref(), 0);
            }
        }
    }
}

/// Scratch bytes required by [`glwe_accumulate_prepared_baby_steps_dft`].
pub(super) fn glwe_accumulate_prepared_baby_steps_dft_tmp_bytes<M>(
    module: &M,
    cnv_offset_hi: usize,
    baby_size: usize,
    diagonal_size: usize,
) -> usize
where
    M: ModuleN + VecZnxDftBytesOf,
{
    let res_dft_size = baby_size + diagonal_size - cnv_offset_hi;
    module.bytes_of_vec_znx_dft(1, res_dft_size)
}
