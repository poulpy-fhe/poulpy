//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/linear_transformation.md: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxDft`.
//! Both flavors hand the giant step to `cnv_accumulate_dft_columns`, which sums
//! the terms over all output columns at once (the diagonal is broadcast across
//! them), so no per-term result is ever materialized.

use poulpy_hal::layouts::CnvPVecLToBackendRef;
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic},
    layouts::{Backend, CnvDftAccTerm, CnvDftStore, CnvPVecRToBackendRef, ScratchArena, VecZnxDftBackendMut},
};

use crate::{
    LinearTransformationGiantStep,
    default::operations::msb_mask_bottom_limb,
    layouts::IntPolyInfos,
    layouts::{GLWEInfos, GLWEToBackendRef, prepared::PreparedDiagonal},
};

use super::LinearTransformationBabySteps;

/// PROD block for one giant step of a resident (prepared) transform, kept in DFT
/// domain.
///
/// The diagonals are already in convolution domain, so the whole giant step is
/// handed to the backend as a single fused multi-column accumulation. Leaves the
/// `(r+1)` output columns in `prod_dft`.
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
    let terms = prepared_giant_terms(cnv_offset_hi, prod_dft, lhs, gs);
    module.cnv_accumulate_dft_columns(cnv_offset_hi, CnvDftStore::Overwrite, prod_dft, 0, cols, &terms, scratch);
}

/// Validates one giant step against its destination and builds its term list.
///
/// One term list for the whole giant step: the diagonal (RHS) is broadcast
/// across the `cols` output columns, so the backend sees every column of every
/// term in a single call. The returned terms borrow only `lhs` and `gs`.
fn prepared_giant_terms<'t, BE: Backend>(
    cnv_offset_hi: usize,
    prod_dft: &VecZnxDftBackendMut<'_, BE>,
    lhs: &'t LinearTransformationBabySteps<BE>,
    gs: &'t LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>,
) -> Vec<CnvDftAccTerm<'t, BE>> {
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

    gs.diagonals
        .iter()
        .map(|d| {
            let diagonal = d.plaintext.cnv();
            let baby = lhs.baby_step(d.baby);
            assert_eq!(baby.cols(), cols);
            assert_eq!(baby.size() + diagonal.size() - cnv_offset_hi, res_dft_size);
            CnvDftAccTerm {
                a: baby.to_backend_ref(),
                a_col: 0,
                b: diagonal.to_backend_ref(),
                b_col: 0,
            }
        })
        .collect()
}

/// Batched [`glwe_accumulate_prepared_baby_steps_dft`]: runs every giant step's
/// PROD block in one HAL call, so a backend can share a prepared baby step
/// appearing in several of them.
///
/// The term sets stay independent, so the giants' baby subsets, diagonal order
/// and duplicate babies are all preserved as-is. Every set is validated and
/// built before any slice of it is taken, and nothing borrowed outlives the call.
pub(super) fn glwe_accumulate_prepared_baby_steps_dft_batch<BE, M>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dfts: &mut [VecZnxDftBackendMut<'_, BE>],
    lhs: &LinearTransformationBabySteps<BE>,
    giant_steps: &[&LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>],
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: Convolution<BE>,
{
    assert_eq!(prod_dfts.len(), giant_steps.len());
    if giant_steps.is_empty() {
        return;
    }
    let cols = lhs.cols();
    let terms: Vec<Vec<CnvDftAccTerm<'_, BE>>> = prod_dfts
        .iter()
        .zip(giant_steps)
        .map(|(prod_dft, gs)| prepared_giant_terms(cnv_offset_hi, prod_dft, lhs, gs))
        .collect();
    let term_sets: Vec<&[CnvDftAccTerm<'_, BE>]> = terms.iter().map(Vec::as_slice).collect();
    module.cnv_accumulate_dft_columns_batch(cnv_offset_hi, CnvDftStore::Overwrite, prod_dfts, 0, cols, &term_sets, scratch);
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
    P: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos,
{
    let cols = lhs.cols();
    let first = gs
        .diagonals
        .first()
        .expect("streamed linear transformation giant step has no diagonals");
    let pt_base2k = first.plaintext.base2k().as_usize();
    // The streamed diagonal is an integer poly encoded across its full physical
    // width, so mask/size use `max_k`/`max_size`, not the (possibly smaller)
    // effective `k`/`size`.
    let pt_k = first.plaintext.encoded_k().as_usize();
    let diagonal_size = first.plaintext.max_size();
    let mask = msb_mask_bottom_limb(pt_base2k, pt_k);
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

        let terms = [CnvDftAccTerm {
            a: baby.to_backend_ref(),
            a_col: 0,
            b: diagonal.to_backend_ref(),
            b_col: 0,
        }];
        let store = if term_idx == 0 {
            CnvDftStore::Overwrite
        } else {
            CnvDftStore::Accumulate
        };
        module.cnv_accumulate_dft_columns(cnv_offset_hi, store, prod_dft, 0, cols, &terms, &mut scratch_1.borrow());
    }
}
