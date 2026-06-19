//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/lt_bsgs.md §6.3: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxDft`.
//! The first term overwrites each output column via `cnv_apply_dft` (which also
//! zeroes the limbs past the convolution bound); the remaining terms accumulate
//! in place with `cnv_apply_dft_accumulate`, so no per-term result is ever
//! materialized.

use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic},
    layouts::{Backend, CnvDftAccTerm, CnvPVecLToBackendRef, CnvPVecRToBackendRef, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    LinearTransformationGiantStep,
    default::operations::msb_mask_bottom_limb,
    layouts::{GLWEInfos, GLWEToBackendRef, prepared::PreparedDiagonal},
};

use super::LinearTransformationBabySteps;

/// PROD block for one giant step of a resident (prepared) transform, kept in DFT
/// domain.
///
/// The diagonals are already in convolution domain, so the whole giant step is
/// handed to the backend as a single fused accumulation per output column: it
/// sums all `baby ⊛ diagonal` terms with one reduction per output limb and
/// writes `prod_dft` once. Leaves the `(r+1)` output columns in `prod_dft`.
pub(super) fn glwe_accumulate_prepared_baby_steps_dft<BE, M>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationBabySteps<BE>,
    gs: &LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: Convolution<BE>,
{
    let cols = lhs.cols();
    let diagonal_size = gs
        .diagonals
        .first()
        .expect("prepared linear transformation giant step has no diagonals")
        .plaintext
        .cnv()
        .size();
    let res_dft_size = lhs.size() + diagonal_size - cnv_offset_hi;
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    for col in 0..cols {
        let terms: Vec<CnvDftAccTerm<'_, BE>> = gs
            .diagonals
            .iter()
            .map(|d| {
                let diagonal = d.plaintext.cnv();
                let baby = lhs.baby_step(d.baby);
                assert_eq!(baby.cols(), cols);
                assert_eq!(baby.size() + diagonal.size() - cnv_offset_hi, res_dft_size);
                CnvDftAccTerm {
                    a: baby.to_backend_ref(),
                    a_col: col,
                    b: diagonal.to_backend_ref(),
                    b_col: 0,
                }
            })
            .collect();
        module.cnv_accumulate_dft(cnv_offset_hi, prod_dft, col, &terms, scratch);
    }
}

/// Scratch bytes required by [`glwe_accumulate_prepared_baby_steps_dft`].
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
    let res_dft_size = baby_size + diagonal_size - cnv_offset_hi;
    module.cnv_accumulate_dft_tmp_bytes(cnv_offset_hi, res_dft_size, baby_size, diagonal_size)
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
    lhs: &LinearTransformationBabySteps<BE>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
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

    // One reused right-operand slot: the whole RHS streams through it.
    let (mut diagonal, mut scratch_1) = scratch.borrow().take_cnv_pvec_right_scratch(module, 1, diagonal_size);

    // Baby is the outer loop, so the first baby initializes every output column
    // (overwrite) and the rest accumulate in place; each diagonal is prepared once.
    for (term_idx, d) in gs.diagonals.iter().enumerate() {
        let baby = lhs.baby_step(d.baby);
        assert_eq!(baby.cols(), cols);
        assert_eq!(baby.size() + diagonal_size - cnv_offset_hi, res_dft_size);

        // Stream the RHS: prepare this diagonal on the fly, then reuse the slot.
        {
            let plaintext = d.plaintext.to_backend_ref();
            module.cnv_prepare_right(&mut diagonal, &plaintext.data, mask, &mut scratch_1.borrow());
        }

        for col in 0..cols {
            if term_idx == 0 {
                module.cnv_apply_dft(
                    cnv_offset_hi,
                    prod_dft,
                    col,
                    &baby.to_backend_ref(),
                    col,
                    &diagonal.to_backend_ref(),
                    0,
                    &mut scratch_1.borrow(),
                );
            } else {
                module.cnv_apply_dft_accumulate(
                    cnv_offset_hi,
                    prod_dft,
                    col,
                    &baby.to_backend_ref(),
                    col,
                    &diagonal.to_backend_ref(),
                    0,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}
