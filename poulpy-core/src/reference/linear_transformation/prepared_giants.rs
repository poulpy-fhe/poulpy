//! Prepared giant-step evaluation.
//!
//! Main loop from docs/guides/linear_transformation.md: each giant bucket first runs PROD (a
//! DFT-domain inner product over prepared baby rotations and prepared diagonals),
//! then ROT rotates that bucket and folds it into a DFT accumulator. The lazy
//! path keeps body add, DFT automorphism, and cross-giant accumulation in DFT;
//! only mask preparation uses scratch BIG/SMALL before key-switching. Incompatible
//! bases fall back to the regular normalized GLWE automorphism path.

use poulpy_hal::api::VecZnxIdftNormalize as _;
use poulpy_hal::layouts::Module;
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxGadgetDecomposeTmpBytes,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalizeTmpBytes, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, PrepareHint, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, vec_znx_dft_backend_ref_from_mut,
    },
};

use std::collections::BTreeMap;

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation, LinearTransformationGiantStep, ScratchArenaTakeCore,
    api::GLWENormalize,
    gadget::{cnv_offset_to_limb_offset, prod_dft_size},
    layouts::{
        DiagonalProd, GGLWEInfos, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        LinearTransformationPlan, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, PreparedDiagonal},
    },
    reference::{
        keyswitching::{GGLWEProductReference, GLWEKeyswitchInternal},
        linear_transformation::{
            inner_product::{
                glwe_accumulate_prepared_baby_steps_dft, glwe_accumulate_prepared_baby_steps_dft_tmp_bytes,
                glwe_accumulate_unprepared_baby_steps_dft,
            },
            lazy::{
                glwe_dft_add_dft_assign, glwe_dft_copy_dft, glwe_idft_dft_into_big, glwe_lazy_giant_automorphism_from_dft,
                glwe_lazy_giant_automorphism_from_dft_tmp_bytes, glwe_lazy_giant_operand, glwe_lazy_giant_operand_size,
                glwe_normalize_big_into,
            },
        },
    },
};

use super::LinearTransformationBabySteps;
use crate::layouts::{GiantTreeLevel, operand_galois_element};

impl<BE: Backend> DiagonalProd<BE> for PreparedDiagonal<BE::OwnedBuf, BE> {
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        glwe_accumulate_prepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
    }

    /// The backend's [`LinearTransformationImpl::glwe_giant_step_batch`](crate::oep::LinearTransformationImpl::glwe_giant_step_batch).
    fn giant_step_batch(module: &Module<BE>) -> usize
    where
        BE: crate::oep::LinearTransformationImpl,
    {
        BE::glwe_giant_step_batch(module)
    }

    /// The backend's [`LinearTransformationImpl::glwe_accumulate_giant_steps`](crate::oep::LinearTransformationImpl::glwe_accumulate_giant_steps).
    fn accumulate_giant_prods(
        module: &Module<BE>,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &LinearTransformationBabySteps<BE>,
        steps: &[LinearTransformationGiantStep<Self>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: crate::oep::LinearTransformationImpl,
        Module<BE>: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        BE::glwe_accumulate_giant_steps(module, cnv_offset_hi, prods, lhs, steps, scratch);
    }
}

/// Reference streamed PROD, exposed for the scheme layer to wire into its
/// [`DiagonalProd`] impl for its own plaintext diagonal type.
pub fn glwe_accumulate_streamed_baby_steps_dft<BE, M, P>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationBabySteps<BE>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    P: GLWEToBackendRef<BE> + crate::layouts::IntPolyInfos + GLWEInfos,
{
    glwe_accumulate_unprepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
}

/// The shared Phase B/C driver: runs the BSGS giant-step loop and finalizes.
///
/// Generic over the diagonal type `P: DiagonalProd`, so the *same* loop drives
/// both the resident transform (`P = PreparedDiagonal`) and the streamed
/// unprepared transform (`P` a plaintext); only the per-giant PROD block
/// ([`DiagonalProd::accumulate_giant_prod`]) differs. Implements the giant-step
/// products, rotations, and final normalization of docs/guides/linear_transformation.md.
///
/// For each giant step `j` it computes `PROD = Σ_k ũ_{j,k} ⊙ rot(v,k)` in DFT,
/// then rotates by `n1·j` and folds the result into a single accumulator. One of
/// two strategies is chosen up front from the
/// `base2k` agreement of `res`, the PROD output, and the keys:
///
/// - **Lazy DFT path** (hot path; bases agree, or no giant rotation): body add,
///   giant automorphism, and cross-giant accumulation all stay in `VecZnxDft`.
///   The only IDFT and the only normalize happen once, at the end. `j == 0`
///   skips ROT entirely.
/// - **Fallback path** (base mismatch): PROD is still in DFT, but each giant
///   contribution is normalized to SMALL and rotated with the public normalized
///   `glwe_automorphism`. Correct, but gives up the lazy savings.
///
/// The terms `(babies[t], lts[t])` share the first one's giant schedule and
/// diagonal layout, checked by the callers with
/// [`assert_linear_transformation_into`](crate::layouts::assert_linear_transformation_into) and
/// [`assert_linear_transformation_sum`](crate::layouts::assert_linear_transformation_sum): each giant step's PROD is
/// summed over them by `sum_giant_prod` before its single rotation, so one
/// pair is the single transform.
///
/// Writes the encryption of `Σ_t M_t·v_t` into `res`.
pub(super) fn glwe_eval_giant_steps<BE, R, P, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: GLWEAutomorphism<BE>
        + GaloisElement
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + ModuleN
        + GGLWEProductReference<BE>
        + GLWEKeyswitchInternal<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + GLWEMulPlain<BE>
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    H: GetAutomorphismKey<BE>,
{
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();
    let res_k = res.k().as_usize();
    let (lhs, rhs) = (babies[0], lts[0]);

    // PROD writes its result in the diagonals' base2k.
    let first_diagonal = &rhs.giant_steps[0].diagonals[0].plaintext;
    let prod_base2k = first_diagonal.base2k();
    let baby_size = lhs.size();
    let diagonal_size = first_diagonal.size();
    let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, prod_base2k.as_usize());
    let prod_size = crate::gadget::prod_dft_size(baby_size, diagonal_size, cnv_offset_hi);

    if let Some((leaves, levels)) = rhs.rotations.giant_tree(rhs.giant_steps.iter().map(|gs| gs.rot)) {
        let tree = GiantTree {
            leaves: &leaves,
            levels: &levels,
            prod_size,
            cnv_offset_hi,
            cnv_offset_lo,
        };
        return glwe_eval_giant_tree(module, &tree, res, babies, lts, keys, scratch);
    }

    let nonzero_giant_rotations = rhs.giant_steps.iter().filter(|gs| gs.rot != 0).count();
    let has_nonzero_giant_rotation = nonzero_giant_rotations != 0;
    // Giant rotations rotate the post-product destination, so their keys are
    // the ones the destination's precision asks for. Every rotating step's key
    // is resolved once, indexed by step, because the steps need not share a
    // layout: the lazy route is only taken when all of them carry the
    // destination's base2k, and the accumulator is sized by the widest key
    // (each key-switch then clamps to its own width). An identity-only
    // transform legitimately resolves no key at all.
    let giant_keys: Vec<Option<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>> = rhs
        .giant_steps
        .iter()
        .map(|gs| {
            (gs.rot != 0).then(|| {
                keys.get_automorphism_key(operand_galois_element(module, res.n().as_usize(), gs.rot), res.k())
                    .unwrap_or_else(|e| panic!("giant-step rotation {}: {e}", gs.rot))
            })
        })
        .collect();
    // No giant rotation: BIG-flow accumulator is always valid (no key required).
    let use_lazy_giant_rotation = giant_keys
        .iter()
        .flatten()
        .all(|key| res_base2k == key.base2k() && prod_base2k == key.base2k());
    // PROD may carry a partial low limb beyond the destination's live
    // precision. Preserve the same guard below that wider input while the
    // sizing helper applies the key's work-region cap.
    let key_output_size = giant_keys
        .iter()
        .flatten()
        .map(|key| {
            crate::gadget::gglwe_product_accumulation_output_size_with_tail(
                res,
                res,
                key,
                nonzero_giant_rotations,
                prod_size.saturating_sub(res.size()),
            )
        })
        .max()
        .unwrap_or(res.size());
    let use_final_lazy_accumulator = !has_nonzero_giant_rotation || use_lazy_giant_rotation;
    let lazy_size = if use_lazy_giant_rotation {
        key_output_size.max(prod_size)
    } else {
        res.size().max(prod_size)
    };

    let scratch = scratch.borrow();

    if use_final_lazy_accumulator {
        // Lazy path: PROD, body add, giant automorphism, and cross-giant
        // accumulation all stay in DFT. The only IDFT is the final one before
        // the single BIG -> SMALL normalization.
        // The products of `batch` giant steps are held at once, for a backend
        // that accumulates several per pass over the baby steps.
        let batch = P::giant_step_batch(module).min(giant_keys.len());
        let (mut prods, scratch_phase) = scratch.take_vec_znx_dft_slice_scratch(res.n().as_usize(), batch, cols, prod_size);
        let (mut lazy_acc_dft, scratch_phase) = scratch_phase.take_vec_znx_dft_scratch(res.n().as_usize(), cols, lazy_size);
        let (mut buckets, mut scratch_phase) =
            take_sum_buckets(scratch_phase, babies.len(), batch, res.n().as_usize(), cols, prod_size);
        for col in 0..cols {
            module.vec_znx_dft_zero(&mut lazy_acc_dft, col);
        }

        let mut res_initialized = false;
        let mut prods: Vec<VecZnxDftBackendMut<'_, BE>> = prods.iter_mut().map(|prod| prod.to_backend_mut()).collect();
        let mut buckets: Vec<VecZnxDftBackendMut<'_, BE>> = buckets.iter_mut().map(|bucket| bucket.to_backend_mut()).collect();
        for (g, giant_key) in giant_keys.iter().enumerate() {
            if g % batch == 0 {
                let steps = g..(g + batch).min(giant_keys.len());
                sum_giant_prods(
                    module,
                    cnv_offset_hi,
                    &mut prods[..steps.len()],
                    &mut buckets,
                    babies,
                    lts,
                    steps,
                    &mut scratch_phase,
                );
            }
            let prod_dft = &mut prods[g % batch];
            let rot = rhs.giant_steps[g].rot;
            if rot == 0 {
                let prod_dft_ref = poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut(prod_dft);
                let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
                if res_initialized {
                    glwe_dft_add_dft_assign(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                } else {
                    glwe_dft_copy_dft(module, &mut lazy_acc_dft_backend, &prod_dft_ref);
                }
            } else {
                let key = giant_key.as_ref().unwrap();
                {
                    let mut lazy_acc_dft_backend = lazy_acc_dft.to_backend_mut();
                    glwe_lazy_giant_automorphism_from_dft(
                        module,
                        &mut lazy_acc_dft_backend,
                        prod_dft,
                        prod_base2k.as_usize(),
                        operand_galois_element(module, res.n().as_usize(), rot),
                        &key.key,
                        key_output_size,
                        nonzero_giant_rotations,
                        res_initialized,
                        &mut scratch_phase,
                    );
                }
            }
            res_initialized = true;
        }
        assert!(res_initialized, "linear transformation has no giant steps");

        let (mut lazy_acc_big, mut scratch_phase) = scratch_phase.take_vec_znx_big_scratch(res.n().as_usize(), cols, lazy_size);
        {
            let mut lazy_acc_big_backend = lazy_acc_big.to_backend_mut();
            glwe_idft_dft_into_big(
                module,
                &mut lazy_acc_big_backend,
                &lazy_acc_dft.to_backend_ref(),
                &mut scratch_phase,
            );
        }

        let lazy_acc_ref = lazy_acc_big.to_backend_ref();
        glwe_normalize_big_into(
            module,
            res,
            &lazy_acc_ref,
            prod_base2k.as_usize(),
            cnv_offset_lo,
            &mut scratch_phase,
        );
        return;
    }

    // Fallback for incompatible bases: PROD is still computed in DFT, then each
    // column is IDFT'd through a one-column BIG scratch only where it is
    // normalized into the temporary SMALL ciphertext.
    let (mut prod_dft, scratch_phase) = scratch.take_vec_znx_dft_scratch(res.n().as_usize(), cols, prod_size);
    let (mut prod_col_big, mut scratch_phase) = scratch_phase.take_vec_znx_big_scratch(res.n().as_usize(), 1, prod_size);
    let mut fallback_acc: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(res);
    let mut res_initialized = false;

    for (g, giant_key) in giant_keys.iter().enumerate() {
        {
            let mut prod_dft_backend = prod_dft.to_backend_mut();
            sum_giant_prod(
                module,
                cnv_offset_hi,
                &mut prod_dft_backend,
                babies,
                lts,
                g,
                &mut scratch_phase,
            );
            let mut acc_backend = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendMut<BE>>::to_backend_mut(&mut fallback_acc);
            let prod_dft_ref = poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut(&prod_dft_backend);
            for col in 0..cols {
                module.vec_znx_idft_apply(&mut prod_col_big, 0, &prod_dft_ref, col, &mut scratch_phase.borrow());
                let prod_col_big_ref = prod_col_big.to_backend_ref();
                module.vec_znx_big_normalize(
                    &mut acc_backend.data,
                    res_base2k.as_usize(),
                    res_k,
                    cnv_offset_lo,
                    col,
                    &prod_col_big_ref,
                    prod_base2k.as_usize(),
                    0,
                    &mut scratch_phase.borrow(),
                );
            }
        }

        if let Some(key) = giant_key.as_ref() {
            module.glwe_automorphism_assign(&mut fallback_acc, key, &mut scratch_phase);
        }

        if res_initialized {
            module.glwe_add_assign(res, &fallback_acc);
        } else {
            module.glwe_copy(res, &fallback_acc, &mut scratch_phase);
            res_initialized = true;
        }
    }

    let mut res_backend = res.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_normalize_assign(
            res_base2k.as_usize(),
            res_k,
            0,
            &mut res_backend.data,
            col,
            &mut scratch_phase.borrow(),
        );
    }
}

/// The giant-step tree of [`crate::layouts::BsgsRotations::PowersOfTwo`] and the product layout.
struct GiantTree<'t> {
    /// The tree node of each giant step's product.
    leaves: &'t [usize],
    levels: &'t [GiantTreeLevel],
    prod_size: usize,
    cnv_offset_hi: usize,
    cnv_offset_lo: i64,
}

/// [`glwe_eval_giant_steps`] for [`crate::layouts::BsgsRotations::PowersOfTwo`]: every product
/// first, then the tree a level at a time, each level switching with one key.
///
/// The lazy route keeps the nodes in DFT, as the per-rotation one does its accumulator; the
/// fallback normalizes the products and runs the tree on ciphertexts. Both hold every
/// product at once, in buffers of their own: the scratch queries take no rotation count.
#[allow(clippy::type_complexity)]
fn glwe_eval_giant_tree<BE, R, P, H>(
    module: &Module<BE>,
    tree: &GiantTree<'_>,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: GLWEAutomorphism<BE>
        + GaloisElement
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + ModuleN
        + GGLWEProductReference<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + VecZnxNormalizeAssign<BE>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + crate::api::GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    H: GetAutomorphismKey<BE>,
{
    let n = res.n().as_usize();
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();
    let prod_base2k = lts[0].giant_steps[0].diagonals[0].plaintext.base2k();
    let steps = lts[0].giant_steps.len();
    let level_keys: Vec<Option<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>> = tree
        .levels
        .iter()
        .map(|level| {
            (!level.pairs.is_empty()).then(|| {
                keys.get_automorphism_key(operand_galois_element(module, n, level.rot), res.k())
                    .unwrap_or_else(|e| panic!("giant-tree rotation {}: {e}", level.rot))
            })
        })
        .collect();
    let switches: usize = tree.levels.iter().map(|level| level.pairs.len()).sum();
    let lazy = level_keys
        .iter()
        .flatten()
        .all(|key| res_base2k == key.base2k() && prod_base2k == key.base2k());

    let diagonal_size = lts[0].giant_steps[0].diagonals[0].plaintext.size();
    let batch = P::giant_step_batch(module).min(steps);
    let shape = GiantTreeShape {
        cnv_offset_hi: tree.cnv_offset_hi,
        baby_size: babies[0].size(),
        diagonal_size,
        prod_size: tree.prod_size,
        res_size: res.size(),
        terms: babies.len(),
        batch,
    };

    if !lazy {
        // The products, then each as a normalized ciphertext, carved from the scratch beside the
        // work of the tree ([`glwe_eval_giant_tree_tmp_bytes`]).
        let need = steps * giant_tree_fallback_buffer_bytes::<BE, _>(module, n, cols, &shape)
            + level_keys
                .iter()
                .flatten()
                .map(|key| giant_tree_switch_bytes::<BE, _, _>(module, res, key, false, 0))
                .fold(giant_tree_work_bytes::<BE, _, _>(module, res, false, &shape, 0), usize::max);
        assert!(
            scratch.available() >= need,
            "glwe_eval_linear_transformation: scratch below its query"
        );
        let (mut prods, scratch_1) = scratch
            .borrow()
            .take_vec_znx_dft_slice_scratch(n, steps, cols, tree.prod_size);
        let (mut leaves, mut scratch_1) = scratch_1.take_glwe_slice_scratch(steps, res);
        {
            let (mut buckets, mut scratch_2) = take_sum_buckets(scratch_1.borrow(), babies.len(), batch, n, cols, tree.prod_size);
            let mut buckets: Vec<VecZnxDftBackendMut<'_, BE>> =
                buckets.iter_mut().map(|bucket| bucket.to_backend_mut()).collect();
            let mut views: Vec<VecZnxDftBackendMut<'_, BE>> = prods.iter_mut().map(|p| p.to_backend_mut()).collect();
            for start in (0..steps).step_by(batch) {
                let range = start..(start + batch).min(steps);
                sum_giant_prods(
                    module,
                    tree.cnv_offset_hi,
                    &mut views[range.clone()],
                    &mut buckets,
                    babies,
                    lts,
                    range,
                    &mut scratch_2,
                );
            }
        }
        // Each product is read no more: one inverse transform and normalization per column.
        for (prod, leaf) in prods.iter_mut().zip(leaves.iter_mut()) {
            let mut leaf_backend = leaf.to_backend_mut();
            let mut prod_backend = prod.to_backend_mut();
            for col in 0..cols {
                module.vec_znx_idft_normalize(
                    &mut leaf_backend.data,
                    res_base2k.as_usize(),
                    res.k().as_usize(),
                    tree.cnv_offset_lo,
                    col,
                    &mut prod_backend,
                    col,
                    prod_base2k.as_usize(),
                    None,
                    &mut scratch_1.borrow(),
                );
            }
        }
        // The odd node of each pair rotated in place and added to the even one.
        let mut nodes: BTreeMap<usize, usize> = tree.leaves.iter().enumerate().map(|(g, &leaf)| (leaf, g)).collect();
        for (level, key) in tree.levels.iter().zip(&level_keys) {
            for &(even, odd) in &level.pairs {
                let odd_slot = nodes.remove(&odd).unwrap();
                module.glwe_automorphism_assign(&mut leaves[odd_slot], key.as_ref().unwrap(), &mut scratch_1);
                match even.and_then(|even| nodes.get(&even).copied()) {
                    Some(even_slot) => {
                        let (even, odd) = two_buffers(&mut leaves, even_slot, odd_slot);
                        module.glwe_add_assign(even, odd);
                    }
                    None => {
                        nodes.insert(odd - 1, odd_slot);
                    }
                }
            }
            nodes = nodes.into_iter().map(|(index, slot)| (index >> 1, slot)).collect();
        }
        // The root normalized into `res` in one pass.
        module.glwe_normalize(res, &leaves[nodes[&0]], &mut scratch_1);
        return;
    }

    let key_output_size = level_keys
        .iter()
        .flatten()
        .map(|key| {
            crate::gadget::gglwe_product_accumulation_output_size_with_tail(
                res,
                res,
                key,
                switches,
                tree.prod_size.saturating_sub(res.size()),
            )
        })
        .max()
        .unwrap_or(res.size());
    let node_size = key_output_size.max(tree.prod_size);
    // A product is stored in a buffer as wide as an accumulator when the layout can view its
    // limbs apart; it is read over its own limbs either way.
    let wide = BE::DFT_LIMBS_CONTIGUOUS && node_size > tree.prod_size;

    // The switches of every level, `(dst, src, accumulate, copy)` over buffers numbered
    // products first: a pair adds into its even node when that is as wide as an accumulator,
    // else into an accumulator of its own, the even node copied there first.
    let mut sizes = vec![if wide { node_size } else { tree.prod_size }; steps];
    let mut zero_tail = vec![false; steps];
    let mut nodes: BTreeMap<usize, usize> = tree.leaves.iter().enumerate().map(|(g, &leaf)| (leaf, g)).collect();
    let mut plan: Vec<Vec<(usize, usize, bool, Option<usize>)>> = Vec::with_capacity(tree.levels.len());
    for level in tree.levels {
        let mut switches_of_level = Vec::with_capacity(level.pairs.len());
        for &(even, odd) in &level.pairs {
            let src = nodes[&odd];
            let switch = match even.map(|even| nodes[&even]) {
                Some(even) if sizes[even] >= node_size => {
                    // A product the pair accumulates into, past which the buffer is zeroed.
                    if wide && even < steps {
                        zero_tail[even] = true;
                    }
                    (even, src, true, None)
                }
                even => {
                    sizes.push(node_size);
                    (sizes.len() - 1, src, even.is_some(), even)
                }
            };
            nodes.insert(odd - 1, switch.0);
            switches_of_level.push(switch);
        }
        plan.push(switches_of_level);
        nodes = nodes
            .into_iter()
            .filter(|(index, _)| index & 1 == 0)
            .map(|(index, buffer)| (index >> 1, buffer))
            .collect();
    }
    let root = nodes[&0];
    // Read as a source over the limbs it held before the level that reads it.
    let mut held = vec![tree.prod_size; steps];
    held.resize(sizes.len(), node_size);

    // Every node and the operands of the widest level in one allocation, so that a backend
    // can take the switches of a level in one pass over their key.
    let operand_size = |key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, limbs: usize| {
        glwe_lazy_giant_operand_size(key, limbs, key_output_size.min(key.size()))
    };
    let operand_bytes = {
        let mut held = held.clone();
        let mut widest = 0;
        for (switches, key) in plan.iter().zip(&level_keys) {
            if let Some(key) = key {
                let bytes: usize = switches
                    .iter()
                    .map(|&(_, src, _, _)| {
                        giant_tree_operand_bytes::<BE, _, _>(module, n, cols, key, held[src], key_output_size.min(key.size()))
                    })
                    .sum();
                widest = widest.max(bytes);
            }
            for &(dst, ..) in switches {
                held[dst] = node_size;
            }
        }
        widest
    };

    let node_bytes: usize = sizes
        .iter()
        .map(|&size| BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, size)))
        .sum();
    // Carved from the scratch beside the work of the tree ([`glwe_eval_giant_tree_tmp_bytes`]).
    let need = node_bytes
        + operand_bytes
        + level_keys
            .iter()
            .flatten()
            .map(|key| giant_tree_switch_bytes::<BE, _, _>(module, res, key, true, node_size))
            .fold(
                giant_tree_work_bytes::<BE, _, _>(module, res, true, &shape, node_size),
                usize::max,
            );
    assert!(
        scratch.available() >= need,
        "glwe_eval_linear_transformation: scratch below its query"
    );
    let mut arena = scratch.borrow();
    let mut buffers = Vec::with_capacity(sizes.len());
    for &size in &sizes {
        let (buffer, rest) = arena.take_vec_znx_dft_scratch(n, cols, size);
        buffers.push(buffer);
        arena = rest;
    }

    {
        let mut full: Vec<VecZnxDftBackendMut<'_, BE>> = buffers[..steps].iter_mut().map(|p| p.to_backend_mut()).collect();
        {
            let (mut buckets, mut products_arena) =
                take_sum_buckets(arena.borrow(), babies.len(), batch, n, cols, tree.prod_size);
            let mut buckets: Vec<VecZnxDftBackendMut<'_, BE>> =
                buckets.iter_mut().map(|bucket| bucket.to_backend_mut()).collect();
            let mut views: Vec<VecZnxDftBackendMut<'_, BE>> =
                full.iter_mut().map(|p| p.with_limb_range_mut(0, tree.prod_size)).collect();
            for start in (0..steps).step_by(batch) {
                let range = start..(start + batch).min(steps);
                sum_giant_prods(
                    module,
                    tree.cnv_offset_hi,
                    &mut views[range.clone()],
                    &mut buckets,
                    babies,
                    lts,
                    range,
                    &mut products_arena,
                );
            }
        }
        // The limbs past the product of a buffer accumulated in place.
        for (leaf, view) in full.iter_mut().enumerate() {
            if zero_tail[leaf] {
                let mut tail = view.with_limb_range_mut(tree.prod_size, node_size);
                for col in 0..cols {
                    module.vec_znx_dft_zero(&mut tail, col);
                }
            }
        }
    }

    for ((level, switches_of_level), key) in tree.levels.iter().zip(&plan).zip(&level_keys) {
        let Some(key) = key else { continue };
        let output_size = key_output_size.min(key.size());
        for &(dst, _, _, copy) in switches_of_level {
            if let Some(even) = copy {
                let (dst, even) = two_buffers(&mut buffers, dst, even);
                glwe_dft_copy_dft(module, &mut dst.to_backend_mut(), &even.to_backend_ref());
            }
        }
        // The destinations and sources of the level, in switch order, each source over the
        // limbs it holds.
        let mut dsts: Vec<Option<VecZnxDftBackendMut<'_, BE>>> = (0..switches_of_level.len()).map(|_| None).collect();
        let mut srcs: Vec<Option<VecZnxDftBackendMut<'_, BE>>> = (0..switches_of_level.len()).map(|_| None).collect();
        for (b, buffer) in buffers.iter_mut().enumerate() {
            if let Some(i) = switches_of_level.iter().position(|&(dst, ..)| dst == b) {
                dsts[i] = Some(buffer.to_backend_mut());
            } else if let Some(i) = switches_of_level.iter().position(|&(_, src, ..)| src == b) {
                srcs[i] = Some(buffer.to_backend_mut());
            }
        }
        let mut dsts: Vec<VecZnxDftBackendMut<'_, BE>> = dsts.into_iter().map(Option::unwrap).collect();
        let mut srcs: Vec<VecZnxDftBackendMut<'_, BE>> = srcs.into_iter().map(Option::unwrap).collect();
        let mut srcs: Vec<VecZnxDftBackendMut<'_, BE>> = srcs
            .iter_mut()
            .zip(switches_of_level)
            .map(|(src, &(_, b, _, _))| src.with_limb_range_mut(0, held[b]))
            .collect();
        // The operands of the level, then the work of its switches past them. Laying out an
        // operand consumes the mask columns of its source, read no more.
        let mut level_arena = arena.borrow();
        let mut operands = Vec::with_capacity(switches_of_level.len());
        for src in &mut srcs {
            let (mut operand, mut rest) = level_arena.take_vec_znx_dft_scratch(n, cols - 1, operand_size(key, src.size()));
            glwe_lazy_giant_operand(
                module,
                &mut operand.to_backend_mut(),
                src,
                prod_base2k.as_usize(),
                &key.key,
                output_size,
                &mut rest,
            );
            operands.push(operand);
            level_arena = rest;
        }
        crate::layouts::assert_no_row_offset("glwe_eval_linear_transformation", &key.key);
        let srcs: Vec<VecZnxDftBackendRef<'_, BE>> = srcs.iter().map(vec_znx_dft_backend_ref_from_mut).collect();
        let addends: Vec<[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>; 1]> = srcs.iter().map(|src| [Some((src, 0))]).collect();
        let addends: Vec<&[Option<(&VecZnxDftBackendRef<'_, BE>, usize)>]> = addends.iter().map(|addend| &addend[..]).collect();
        let operands: Vec<VecZnxDftBackendRef<'_, BE>> = operands.iter().map(|operand| operand.to_backend_ref()).collect();
        let accumulate: Vec<bool> = switches_of_level.iter().map(|&(_, _, accumulate, _)| accumulate).collect();
        BE::gglwe_products_fused(
            module,
            &mut dsts,
            output_size,
            &operands,
            &key.key,
            switches,
            &addends,
            operand_galois_element(module, n, level.rot),
            &accumulate,
            &mut level_arena,
        );
        drop(dsts);
        for &(dst, ..) in switches_of_level {
            held[dst] = node_size;
        }
    }
    // The root is read no more: one inverse transform and normalization per column, over the
    // limbs it holds.
    let res_k = res.k().as_usize();
    let mut res_backend = res.to_backend_mut();
    let mut root_backend = buffers[root].to_backend_mut();
    let mut root_view = root_backend.with_limb_range_mut(0, held[root]);
    for col in 0..cols {
        module.vec_znx_idft_normalize(
            &mut res_backend.data,
            res_base2k.as_usize(),
            res_k,
            tree.cnv_offset_lo,
            col,
            &mut root_view,
            col,
            prod_base2k.as_usize(),
            None,
            &mut arena.borrow(),
        );
    }
}

/// `buffers[dst]` beside `buffers[src]`, `dst != src`.
fn two_buffers<T>(buffers: &mut [T], dst: usize, src: usize) -> (&mut T, &T) {
    assert_ne!(dst, src);
    if dst < src {
        let (lo, hi) = buffers.split_at_mut(src);
        (&mut lo[dst], &hi[0])
    } else {
        let (lo, hi) = buffers.split_at_mut(dst);
        (&mut hi[0], &lo[src])
    }
}

/// The sizes the giant-step tree lays its scratch out from: a transform's in its evaluation,
/// upper bounds of them in [`glwe_eval_giant_tree_tmp_bytes`].
struct GiantTreeShape {
    cnv_offset_hi: usize,
    /// Limbs of a baby rotation.
    baby_size: usize,
    /// Limbs of a diagonal.
    diagonal_size: usize,
    /// Limbs of a giant step's product.
    prod_size: usize,
    /// Limbs of the destination.
    res_size: usize,
    /// Terms of a sum, one for a single transform.
    terms: usize,
    /// Giant steps whose products are computed together.
    batch: usize,
}

/// Bytes of the buffers of a giant step on the route on ciphertexts: its product and that
/// product normalized.
fn giant_tree_fallback_buffer_bytes<BE, M>(module: &M, n: usize, cols: usize, shape: &GiantTreeShape) -> usize
where
    BE: Backend,
    M: VecZnxDftBytesOf,
{
    BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, shape.prod_size))
        + BE::scratch_aligned(BE::bytes_of_vec_znx(n, cols, shape.res_size))
}

/// Bytes of the work of the giant-step tree beside its buffers that no key sizes: a product,
/// then the normalizations, of the root of `node_size` limbs on the lazy route, of each
/// product and of the root on the route on ciphertexts.
fn giant_tree_work_bytes<BE, M, R>(module: &M, res: &R, lazy: bool, shape: &GiantTreeShape, node_size: usize) -> usize
where
    BE: Backend,
    M: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxIdftNormalizeTmpBytes + GLWENormalize<BE>,
    R: GLWEInfos,
{
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    // A later term of a sum is computed into a bucket per step of a batch, a streamed diagonal prepared into a slot
    // and multiplied a diagonal at a time.
    let bucket = if shape.terms > 1 {
        shape.batch * BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, shape.prod_size))
    } else {
        0
    };
    let prod = bucket
        + BE::scratch_aligned(module.bytes_of_cnv_pvec_right(n, 1, shape.diagonal_size, PrepareHint::OneShot))
        + glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(
            module,
            shape.cnv_offset_hi,
            shape.baby_size,
            shape.diagonal_size,
        )
        .max(module.cnv_apply_dft_tmp_bytes(shape.cnv_offset_hi, shape.prod_size, shape.baby_size, shape.diagonal_size))
        .max(module.cnv_prepare_right_tmp_bytes(shape.diagonal_size, shape.diagonal_size));
    let normalize = if lazy {
        module.vec_znx_idft_normalize_tmp_bytes(shape.res_size, node_size)
    } else {
        module
            .vec_znx_idft_normalize_tmp_bytes(shape.res_size, shape.prod_size)
            .max(module.glwe_normalize_tmp_bytes())
    };
    prod.max(normalize)
}

/// Bytes of the operand of `key` for a node of `limbs` limbs switched over `output_size` limbs,
/// one of the operands a level of the lazy giant-step tree lays out before its switches.
fn giant_tree_operand_bytes<BE, M, K>(module: &M, n: usize, cols: usize, key: &K, limbs: usize, output_size: usize) -> usize
where
    BE: Backend,
    M: VecZnxDftBytesOf,
    K: GGLWEInfos,
{
    BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols - 1, glwe_lazy_giant_operand_size(key, limbs, output_size)))
}

/// Bytes of a switch of the giant-step tree with `key`: of a node of `node_size` limbs on the
/// lazy route, of a ciphertext of the layout of `res` on the other.
fn giant_tree_switch_bytes<BE, R, K>(module: &Module<BE>, res: &R, key: &K, lazy: bool, node_size: usize) -> usize
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl,
    Module<BE>: ModuleN
        + GLWEAutomorphism<BE>
        + VecZnxBigBytesOf
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    if lazy {
        glwe_lazy_giant_automorphism_from_dft_tmp_bytes::<BE, _>(module, res.rank().as_usize(), node_size, key)
    } else {
        module.glwe_automorphism_tmp_bytes(res, res, key)
    }
}

/// Bytes the giant-step tree of a transform on [`crate::layouts::BsgsRotations::PowersOfTwo`]
/// with the schedule `plan` takes from the scratch, the larger of its two routes: `res` is the
/// destination, `a` the source, `pt` a diagonal and `key` the widest giant key. Zero for
/// [`crate::layouts::BsgsRotations::PerRotation`].
pub(super) fn glwe_eval_giant_tree_tmp_bytes<BE, R, A, B, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    pt: &B,
    plan: &LinearTransformationPlan,
    key: &K,
) -> usize
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecBytesOf
        + GLWEAutomorphism<BE>
        + GLWENormalize<BE>
        + VecZnxBigBytesOf
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: LWEInfos,
    K: GGLWEInfos,
{
    let Some((_, levels)) = plan.rotations.giant_tree(plan.giant_steps.iter().copied()) else {
        return 0;
    };
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    let steps = plan.giant_steps.len();
    let switches: usize = levels.iter().map(|level| level.pairs.len()).sum();
    // No conversion offset retains the widest product; a key-switch output is no wider than
    // its key, and a tree without one keeps nodes as wide as the destination.
    let shape = GiantTreeShape {
        cnv_offset_hi: 0,
        baby_size: a.max_size(),
        diagonal_size: pt.max_size(),
        prod_size: prod_dft_size(a.max_size(), pt.max_size(), 0),
        res_size: res.max_size(),
        terms: 1,
        batch: BE::glwe_giant_step_batch(module).min(steps),
    };
    let node_size = key.size().max(shape.prod_size).max(shape.res_size);
    // A buffer per giant step and at most one per switch, as wide as a node, and the operands
    // of a level.
    // The operands of the level of most switches, each as wide as a node's.
    let level_operands = levels.iter().map(|level| level.pairs.len()).max().unwrap_or(0)
        * giant_tree_operand_bytes::<BE, _, _>(module, n, cols, key, node_size, key.size());
    let lazy = (steps + switches) * BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, node_size))
        + level_operands
        + giant_tree_work_bytes::<BE, _, _>(module, res, true, &shape, node_size)
            .max(giant_tree_switch_bytes::<BE, _, _>(module, res, key, true, node_size));
    let fallback = steps * giant_tree_fallback_buffer_bytes::<BE, _>(module, n, cols, &shape)
        + giant_tree_work_bytes::<BE, _, _>(module, res, false, &shape, node_size)
            .max(giant_tree_switch_bytes::<BE, _, _>(module, res, key, false, node_size));
    lazy.max(fallback)
}

/// The buckets [`sum_giant_prods`] computes the later terms of a sum into: one per product of a
/// batch of `batch` products of `size` limbs, carved once per evaluation, and none for a single
/// transform.
fn take_sum_buckets<BE: Backend>(
    scratch: ScratchArena<'_, BE>,
    terms: usize,
    batch: usize,
    n: usize,
    cols: usize,
    size: usize,
) -> (Vec<poulpy_hal::layouts::VecZnxDftViewMut<'_, BE>>, ScratchArena<'_, BE>) {
    scratch.take_vec_znx_dft_slice_scratch(n, if terms > 1 { batch } else { 0 }, cols, size)
}

/// [`sum_giant_prod`] of the giant steps `steps`, `prods[i]` receiving step
/// `steps.start + i`, through [`DiagonalProd::accumulate_giant_prods`]: the
/// first term writes the products, each later one is computed into the first
/// `prods.len()` of `buckets` ([`take_sum_buckets`]) and added in DFT.
#[allow(clippy::too_many_arguments)]
fn sum_giant_prods<BE, P>(
    module: &Module<BE>,
    cnv_offset_hi: usize,
    prods: &mut [VecZnxDftBackendMut<'_, BE>],
    buckets: &mut [VecZnxDftBackendMut<'_, BE>],
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    steps: core::ops::Range<usize>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: CnvPVecBytesOf + Convolution<BE> + ModuleN + VecZnxDftAddAssign<BE>,
    P: DiagonalProd<BE>,
{
    P::accumulate_giant_prods(
        module,
        cnv_offset_hi,
        prods,
        babies[0],
        &lts[0].giant_steps[steps.clone()],
        scratch,
    );
    if babies.len() == 1 {
        return;
    }
    let buckets = &mut buckets[..prods.len()];
    for (&lhs, &rhs) in babies[1..].iter().zip(&lts[1..]) {
        P::accumulate_giant_prods(module, cnv_offset_hi, buckets, lhs, &rhs.giant_steps[steps.clone()], scratch);
        for (prod, bucket) in prods.iter_mut().zip(buckets.iter()) {
            glwe_dft_add_dft_assign(module, prod, &poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut(bucket));
        }
    }
}

/// PROD of giant step `g` summed over the terms `(babies[t], lts[t])`: the first term writes
/// `prod_dft`, each later one is computed into one scratch bucket, taken only
/// for a sum, and added in DFT.
fn sum_giant_prod<BE, M, P>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    g: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN + VecZnxDftAddAssign<BE>,
    P: DiagonalProd<BE>,
{
    let (lhs, rhs) = (babies[0], lts[0]);
    P::accumulate_giant_prod(module, cnv_offset_hi, prod_dft, lhs, &rhs.giant_steps[g], scratch);
    if babies.len() == 1 {
        return;
    }
    let (mut bucket, mut scratch_bucket) =
        scratch
            .borrow()
            .take_vec_znx_dft_scratch(prod_dft.n(), prod_dft.cols(), prod_dft.size());
    for (&lhs, &rhs) in babies[1..].iter().zip(&lts[1..]) {
        P::accumulate_giant_prod(
            module,
            cnv_offset_hi,
            &mut bucket.to_backend_mut(),
            lhs,
            &rhs.giant_steps[g],
            &mut scratch_bucket,
        );
        glwe_dft_add_dft_assign(module, prod_dft, &bucket.to_backend_ref());
    }
}
