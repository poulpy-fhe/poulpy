//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/lt_bsgs.md §6.3: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxDft`.
//! `cnv_apply_dft` overwrites its destination, so the first term initializes
//! each DFT column and following terms are accumulated explicitly.

use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftCopy},
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, ScratchArena, VecZnxDftBackendMut, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef,
    },
};

use crate::{
    LinearTransformationGiantStep, LinearTransformationPlan,
    default::operations::msb_mask_bottom_limb,
    layouts::{GLWEInfos, GLWEToBackendRef},
};

use super::{LinearTransformationLhsPrepared, LinearTransformationRhsGiantStepPrepared};

/// PROD block for one giant step, kept in DFT domain.
///
/// Leaves the `(r+1)` output columns in `prod_dft`. Convolution output is staged
/// through a one-column DFT scratch so all backends follow the same
/// column-selection path; the first term is copied into `prod_dft`, and later
/// terms are accumulated with `vec_znx_dft_add_assign`.
pub(super) fn glwe_accumulate_prepared_baby_steps_dft<BE, M>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationLhsPrepared<BE>,
    rhs: &LinearTransformationRhsGiantStepPrepared<BE>,
    plan: &LinearTransformationPlan,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN + Convolution<BE> + VecZnxDftAddAssign<BE> + VecZnxDftBytesOf + VecZnxDftCopy<BE>,
{
    let cols = lhs.cols();
    let res_dft_size = lhs.size() + rhs.size() - cnv_offset_hi;
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    let (mut term_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_dft_size);

    // Baby is the outer loop, so the first baby initializes every output column
    // (copy) and the rest accumulate (add).
    for (term_idx, &baby_step_idx) in rhs.baby_step_indexes().iter().enumerate() {
        let baby_rot = plan.baby_steps[baby_step_idx];
        let diagonal = rhs.diagonal(baby_rot);
        let baby = lhs.baby_step(baby_rot);
        assert_eq!(baby.cols(), cols);
        assert_eq!(baby.size() + diagonal.size() - cnv_offset_hi, res_dft_size);
        for col in 0..cols {
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

/// PROD block for one giant step from an *unprepared* matrix: identical to
/// [`glwe_accumulate_prepared_baby_steps_dft`] except each diagonal is prepared
/// (`cnv_prepare_right`) on the fly into a single reused scratch `CnvPVecR` and
/// discarded immediately, so the full prepared RHS is never materialized. Trades
/// per-eval recompute for memory — for memory-bound backends (e.g. GPU). The
/// baby loop is the outer one so each diagonal is prepared exactly once per eval.
pub(super) fn glwe_accumulate_unprepared_baby_steps_dft<BE, M, P>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationLhsPrepared<BE>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN + VecZnxDftAddAssign<BE> + VecZnxDftBytesOf + VecZnxDftCopy<BE>,
    P: GLWEToBackendRef<BE> + GLWEInfos,
{
    let cols = lhs.cols();
    let first = gs
        .diagonals
        .first()
        .expect("streamed linear transformation giant step has no diagonals");
    let pt_base2k = first.plaintext.base2k().as_usize();
    let pt_max_k = first.plaintext.max_k().as_usize();
    let diagonal_size = first.plaintext.size();
    let mask = msb_mask_bottom_limb(pt_base2k, pt_max_k);
    let res_dft_size = lhs.size() + diagonal_size - cnv_offset_hi;
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    let (mut term_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, 1, res_dft_size);
    // One reused right-operand slot: the whole RHS streams through it.
    let (mut diagonal, mut scratch_2) = scratch_1.take_cnv_pvec_right_scratch(module, 1, diagonal_size);

    // Baby is the outer loop, so the first baby initializes every output column
    // (copy) and the rest accumulate (add); each diagonal is prepared once.
    for (term_idx, d) in gs.diagonals.iter().enumerate() {
        let baby = lhs.baby_step(d.baby);
        assert_eq!(baby.cols(), cols);
        assert_eq!(baby.size() + diagonal_size - cnv_offset_hi, res_dft_size);

        // Stream the RHS: prepare this diagonal on the fly, then reuse the slot.
        {
            let plaintext = d.plaintext.to_backend_ref();
            module.cnv_prepare_right(&mut diagonal, &plaintext.data, mask, &mut scratch_2.borrow());
        }

        for col in 0..cols {
            module.cnv_apply_dft(
                cnv_offset_hi,
                &mut term_dft.to_backend_mut(),
                0,
                &baby.to_backend_ref(),
                col,
                &diagonal.to_backend_ref(),
                0,
                &mut scratch_2.borrow(),
            );

            if term_idx == 0 {
                module.vec_znx_dft_copy(1, 0, prod_dft, col, &term_dft.to_backend_ref(), 0);
            } else {
                module.vec_znx_dft_add_assign(prod_dft, col, &term_dft.to_backend_ref(), 0);
            }
        }
    }
}
