//! Prepared giant-step evaluation.
//!
//! Main loop from docs/guides/linear_transformation.md: each giant bucket first runs PROD (a
//! DFT-domain inner product over prepared baby rotations and prepared diagonals),
//! then ROT rotates that bucket and folds it into a DFT accumulator. The lazy
//! path keeps body add, DFT automorphism, and cross-giant accumulation in DFT;
//! only mask preparation uses scratch BIG/SMALL before key-switching. Incompatible
//! bases fall back to the regular normalized GLWE automorphism path.

use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, PrepareHint, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
};

use std::collections::BTreeMap;

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation, LinearTransformationGiantStep,
    layouts::{
        GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, ModuleCoreAlloc,
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
                glwe_lazy_giant_automorphism_from_dft_tmp_bytes, glwe_normalize_big_into,
            },
        },
        operations::cnv_offset_to_limb_offset,
    },
};

use super::LinearTransformationBabySteps;
use crate::layouts::{GiantTreeLevel, operand_galois_element};

/// Per-giant PROD, provided by the diagonal representation itself.
///
/// The shared evaluator `glwe_eval_giant_steps` is generic over the diagonal
/// type `P` stored in the [`LinearTransformation`]; the only per-flavor step is
/// how one giant step's `Σ_k ũ_{j,k} ⊙ rot(v,k)` is computed. Each concrete
/// diagonal type implements this once:
///
/// - [`PreparedDiagonal`] (resident): diagonals are already in convolution
///   domain, so the whole giant step is one fused accumulation
///   (`glwe_accumulate_prepared_baby_steps_dft`).
/// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
///   through one reused scratch slot
///   (`glwe_accumulate_unprepared_baby_steps_dft`); implemented by the scheme
///   layer for its plaintext type.
///
/// Dispatching per concrete type — rather than via a blanket keyed on
/// [`GLWEToBackendRef`] — is what keeps the impls coherent: the backend type
/// parameter prevents the compiler from ruling out a downstream
/// `GLWEToBackendRef` impl for [`PreparedDiagonal`], so a blanket would clash
/// with the resident impl.
pub trait DiagonalProd<BE: Backend>: LWEInfos + Sized {
    /// Runs the PROD inner product of one giant step into `prod_dft`.
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN;
}

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

/// Panics unless the first term has a giant step, and every giant step of every term has a diagonal as wide as its
/// transform's first diagonal and reads only baby rotations its term's cache holds. The other diagonals are not
/// checked: every diagonal of a transform shares one width, scale and degree, as the encoder produces them, and
/// preparing asserts the degree. Reads no cache layout, so an empty cache fails on its first missing rotation.
pub(super) fn assert_linear_transformation_giant_steps<BE, P>(
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
) where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    assert!(
        !lts[0].giant_steps.is_empty(),
        "linear transformation has no non-empty giant steps"
    );
    for (t, (&lhs, &lt)) in babies.iter().zip(lts).enumerate() {
        let width = lt.first_diagonal_plaintext().map(|d| d.max_size());
        for (g, gs) in lt.giant_steps.iter().enumerate() {
            let first = gs
                .diagonals
                .first()
                .unwrap_or_else(|| panic!("giant step {g} of linear transformation {t} has no diagonals"));
            assert!(
                Some(first.plaintext.max_size()) == width,
                "giant step {g} of linear transformation {t} is not as wide as the first diagonal"
            );
            for d in &gs.diagonals {
                assert!(
                    lhs.contains_baby_step(d.baby),
                    "missing prepared baby-step rotation {} of linear transformation {t}",
                    d.baby
                );
            }
        }
    }
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
/// [`assert_linear_transformation_giant_steps`]: each giant step's PROD is
/// summed over them by `sum_giant_prod` before its single rotation, so one
/// pair is the single transform.
///
/// Writes the encryption of `Σ_t M_t·v_t` into `res`.
pub(super) fn glwe_eval_giant_steps<BE, M, R, P, H>(
    module: &M,
    cnv_offset: usize,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
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
    let prod_size = crate::reference::operations::prod_dft_size(baby_size, diagonal_size, cnv_offset_hi);

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
            crate::reference::keyswitching::gglwe_product_accumulation_output_size_with_tail(
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
        let (mut prod_dft, scratch_phase) = scratch.take_vec_znx_dft_scratch(res.n().as_usize(), cols, prod_size);
        let (mut lazy_acc_dft, mut scratch_phase) = scratch_phase.take_vec_znx_dft_scratch(res.n().as_usize(), cols, lazy_size);
        for col in 0..cols {
            module.vec_znx_dft_zero(&mut lazy_acc_dft, col);
        }

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
            }
            let rot = rhs.giant_steps[g].rot;
            if rot == 0 {
                let prod_dft_ref = prod_dft.to_backend_ref();
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
                        &mut prod_dft.to_backend_mut(),
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
fn glwe_eval_giant_tree<BE, M, R, P, H>(
    module: &M,
    tree: &GiantTree<'_>,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
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
        + VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + VecZnxNormalizeAssign<BE>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
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

    if !lazy {
        // The products, then each as a normalized ciphertext, carved from the scratch
        // ([`glwe_eval_giant_tree_tmp_bytes`]).
        let (mut prods, scratch_1) = scratch
            .borrow()
            .take_vec_znx_dft_slice_scratch(n, steps, cols, tree.prod_size);
        use crate::ScratchArenaTakeCore;
        let (mut leaves, mut scratch_1) = scratch_1.take_glwe_slice_scratch(steps, res);
        for (g, prod) in prods.iter_mut().enumerate() {
            sum_giant_prod(
                module,
                tree.cnv_offset_hi,
                &mut prod.to_backend_mut(),
                babies,
                lts,
                g,
                &mut scratch_1,
            );
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
            crate::reference::keyswitching::gglwe_product_accumulation_output_size_with_tail(
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

    let node_bytes: usize = sizes
        .iter()
        .map(|&size| BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, size)))
        .sum();
    // The work beside the buffers: the products, an operand and its switch, the root.
    let diagonal_size = lts[0].giant_steps[0].diagonals[0].plaintext.size();
    let bucket = if babies.len() > 1 {
        BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, tree.prod_size))
    } else {
        0
    };
    let prods_work =
        glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(module, tree.cnv_offset_hi, babies[0].size(), diagonal_size)
            + module.bytes_of_cnv_pvec_right(n, 1, diagonal_size, PrepareHint::OneShot)
            + module.cnv_prepare_right_tmp_bytes(diagonal_size, diagonal_size)
            + bucket;
    let switch_work = level_keys
        .iter()
        .flatten()
        .map(|key| glwe_lazy_giant_automorphism_from_dft_tmp_bytes::<BE, _, _>(module, cols - 1, node_size, &key.key))
        .max()
        .unwrap_or(0);
    let root_work = BE::scratch_aligned(module.bytes_of_vec_znx_big(n, cols, node_size))
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let work = prods_work.max(switch_work).max(root_work);
    // Carved from the scratch beside that work ([`glwe_eval_giant_tree_tmp_bytes`]).
    assert!(
        scratch.available() >= node_bytes + work,
        "glwe_eval_linear_transformation: scratch below its query"
    );
    let mut arena = scratch.borrow();
    let mut buffers = Vec::with_capacity(sizes.len());
    for &size in &sizes {
        let (buffer, rest) = arena.take_vec_znx_dft_scratch(n, cols, size);
        buffers.push(buffer);
        arena = rest;
    }

    for (g, buffer) in buffers[..steps].iter_mut().enumerate() {
        let mut full = buffer.to_backend_mut();
        sum_giant_prod(
            module,
            tree.cnv_offset_hi,
            &mut full.with_limb_range_mut(0, tree.prod_size),
            babies,
            lts,
            g,
            &mut arena,
        );
        // The limbs past the product of a buffer accumulated in place.
        if zero_tail[g] {
            let mut tail = full.with_limb_range_mut(tree.prod_size, node_size);
            for col in 0..cols {
                module.vec_znx_dft_zero(&mut tail, col);
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
        // Each source over the limbs it holds; laying out its operand consumes its mask
        // columns, read no more.
        for &(dst, src, accumulate, _) in switches_of_level {
            let (dst, src_buffer) = two_buffers_mut(&mut buffers, dst, src);
            let mut src_view = src_buffer.to_backend_mut();
            glwe_lazy_giant_automorphism_from_dft(
                module,
                &mut dst.to_backend_mut(),
                &mut src_view.with_limb_range_mut(0, held[src]),
                prod_base2k.as_usize(),
                operand_galois_element(module, n, level.rot),
                &key.key,
                output_size,
                switches,
                accumulate,
                &mut arena,
            );
        }
        for &(dst, ..) in switches_of_level {
            held[dst] = node_size;
        }
    }
    let root = &buffers[root];
    let (mut root_big, mut scratch_1) = arena.take_vec_znx_big_scratch(n, cols, root.size());
    glwe_idft_dft_into_big(module, &mut root_big.to_backend_mut(), &root.to_backend_ref(), &mut scratch_1);
    glwe_normalize_big_into(
        module,
        res,
        &root_big.to_backend_ref(),
        prod_base2k.as_usize(),
        tree.cnv_offset_lo,
        &mut scratch_1,
    );
}

/// `buffers[dst]` beside `buffers[src]`, both mutable, `dst != src`.
fn two_buffers_mut<T>(buffers: &mut [T], dst: usize, src: usize) -> (&mut T, &mut T) {
    assert_ne!(dst, src);
    if dst < src {
        let (lo, hi) = buffers.split_at_mut(src);
        (&mut lo[dst], &mut hi[0])
    } else {
        let (lo, hi) = buffers.split_at_mut(dst);
        (&mut hi[0], &mut lo[src])
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

/// Bytes the giant-step tree of [`crate::layouts::BsgsRotations::PowersOfTwo`] takes from the
/// scratch for a transform of `giant_steps` giant steps on top of the per-rotation budget, so
/// that it holds its nodes there instead of allocating them: `a` is the source, `pt` a
/// diagonal and `key` the widest giant key.
pub fn glwe_eval_giant_tree_tmp_bytes<BE, A, B, K>(giant_steps: usize, a: &A, pt: &B, key: &K) -> usize
where
    BE: Backend,
    A: GLWEInfos,
    B: LWEInfos,
    K: crate::layouts::GGLWEInfos,
{
    let (n, cols) = (a.n().as_usize(), a.rank().as_usize() + 1);
    let node = (a.max_size() + pt.max_size()).max(key.size());
    2 * giant_steps * BE::scratch_aligned(BE::bytes_of_vec_znx_dft(n, cols, node))
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
