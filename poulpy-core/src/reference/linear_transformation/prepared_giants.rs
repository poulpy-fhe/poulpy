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
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, vec_znx_dft_backend_ref_from_mut,
    },
};

use std::collections::BTreeMap;

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation, LinearTransformationGiantStep, ScratchArenaTakeCore,
    api::GLWENormalize,
    gadget::{cnv_offset_to_limb_offset, prod_dft_size},
    layouts::{
        Base2K, DiagonalOperand, DiagonalProd, GGLWEInfos, GLWE, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos,
        LinearTransformationPlan, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, PreparedDiagonal},
    },
    reference::{
        keyswitching::{GGLWEProductReference, GLWEKeyswitchInternal, glwe::gglwe_product_from_dft},
        linear_transformation::{
            inner_product::glwe_accumulate_prepared_baby_steps_dft_tmp_bytes,
            lazy::{glwe_dft_add_dft_assign, glwe_dft_copy_dft, glwe_idft_dft_into_big, glwe_normalize_big_into},
        },
    },
};

use super::LinearTransformationBabySteps;
use crate::layouts::{GiantTreeLevel, operand_galois_element};

impl<BE: Backend> DiagonalProd<BE> for PreparedDiagonal<BE::OwnedBuf, BE> {
    fn operand(&self) -> DiagonalOperand<'_, BE> {
        DiagonalOperand::Prepared(&self.cnv)
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

/// The products of the giant steps of one evaluation, which [`glwe_eval_giant_steps`] rotates and sums: each
/// `rank + 1` columns in DFT at the first diagonal's base2k on [`Self::size`] limbs, the low
/// [`Self::cnv_offset_lo`] bits of the conversion offset left to the final normalization.
pub(super) trait GiantProducts<BE: Backend> {
    fn size(&self) -> usize;

    fn cnv_offset_lo(&self) -> i64;

    /// Most giant steps [`Self::products`] takes per call, at least 1.
    fn batch(&self, module: &Module<BE>) -> usize;

    /// Bytes [`Self::products`] takes beside the products, for [`Self::batch`] steps of degree `n`.
    fn tmp_bytes(&self, module: &Module<BE>, n: usize) -> usize;

    /// The products of the giant steps `steps`, `prods[i]` receiving step `steps.start + i`.
    fn products(
        &self,
        module: &Module<BE>,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        steps: core::ops::Range<usize>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// The products of the transforms `lts` from the baby caches `babies`, summed over the terms `(babies[t], lts[t])`,
/// in the diagonals' base2k.
struct PlainProducts<'a, BE: Backend, P> {
    babies: &'a [&'a LinearTransformationBabySteps<BE>],
    lts: &'a [&'a LinearTransformation<P>],
    cnv_offset_hi: usize,
    cnv_offset_lo: i64,
}

impl<BE, P> PlainProducts<'_, BE, P>
where
    BE: Backend,
    P: DiagonalProd<BE>,
{
    fn diagonal(&self) -> &P {
        &self.lts[0].giant_steps[0].diagonals[0].plaintext
    }
}

impl<BE, P> GiantProducts<BE> for PlainProducts<'_, BE, P>
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: CnvPVecBytesOf + Convolution<BE> + ModuleN + VecZnxDftAddAssign<BE> + VecZnxDftBytesOf,
    P: DiagonalProd<BE>,
{
    fn size(&self) -> usize {
        prod_dft_size(self.babies[0].size(), self.diagonal().size(), self.cnv_offset_hi)
    }

    fn cnv_offset_lo(&self) -> i64 {
        self.cnv_offset_lo
    }

    fn batch(&self, module: &Module<BE>) -> usize {
        P::giant_step_batch(module)
    }

    fn tmp_bytes(&self, module: &Module<BE>, n: usize) -> usize {
        plain_products_tmp_bytes::<BE, _>(
            module,
            n,
            self.babies[0].cols(),
            &PlainProductsShape {
                cnv_offset_hi: self.cnv_offset_hi,
                baby_size: self.babies[0].size(),
                diagonal_size: self.diagonal().size(),
                prod_size: self.size(),
                terms: self.babies.len(),
                batch: self.batch(module),
            },
        )
    }

    fn products(
        &self,
        module: &Module<BE>,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        steps: core::ops::Range<usize>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let (n, cols, size) = (prods[0].n(), prods[0].cols(), prods[0].size());
        let (mut buckets, mut scratch) = take_sum_buckets(scratch.borrow(), self.babies.len(), prods.len(), n, cols, size);
        let mut buckets: Vec<VecZnxDftBackendMut<'_, BE>> = buckets.iter_mut().map(|bucket| bucket.to_backend_mut()).collect();
        sum_giant_prods(
            module,
            self.cnv_offset_hi,
            prods,
            &mut buckets,
            self.babies,
            self.lts,
            steps,
            &mut scratch,
        );
    }
}

/// The shared Phase B/C driver: runs the BSGS giant-step loop of the schedule of `lts[0]` over the products of
/// [`PlainProducts`] for plaintext diagonals, of [`CiphertextProducts`] for diagonals of the rank of the source, then
/// finalizes. Implements the giant-step rotations and final normalization of docs/guides/linear_transformation.md.
///
/// The terms `(babies[t], lts[t])` share the first one's giant schedule and diagonal layout, checked by the callers
/// with [`assert_linear_transformation_into`](crate::layouts::assert_linear_transformation_into) and
/// [`assert_linear_transformation_sum`](crate::layouts::assert_linear_transformation_sum): each giant step's PROD is
/// summed over them before its single rotation, so one pair is the single transform.
///
/// For each giant step `j` it takes its product `PROD` in DFT,
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
/// Writes the encryption of `Σ_j rot(PROD_j, n1·j)` into `res`.
pub(super) fn glwe_eval_giant_steps<BE, R, P, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: crate::GLWEPreparedTensoring<BE>
        + crate::reference::operations::GLWETensoringReference<BE>
        + crate::api::GLWEBytesOf<BE>
        + GLWEAutomorphism<BE>
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
    H: GetAutomorphismKey<BE> + crate::layouts::GetTensorKey<BE>,
{
    let schedule = lts[0];
    let diagonal = &schedule.giant_steps[0].diagonals[0].plaintext;
    let prod_base2k = diagonal.base2k();
    let (ciphertext_products, plain_products);
    let products: &dyn GiantProducts<BE> = if diagonal.rank().as_usize() > 0 {
        ciphertext_products =
            super::ciphertext_diagonals::CiphertextProducts::new(babies, lts, keys, res.glwe_layout(), cnv_offset);
        &ciphertext_products
    } else {
        let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, prod_base2k.as_usize());
        plain_products = PlainProducts {
            babies,
            lts,
            cnv_offset_hi,
            cnv_offset_lo,
        };
        &plain_products
    };
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();
    let res_k = res.k().as_usize();
    let prod_size = products.size();
    let cnv_offset_lo = products.cnv_offset_lo();

    if let Some((leaves, levels)) = schedule.rotations.giant_tree(schedule.giant_steps.iter().map(|gs| gs.rot)) {
        let tree = GiantTree {
            leaves: &leaves,
            levels: &levels,
        };
        return glwe_eval_giant_tree(module, &tree, res, prod_base2k, products, keys, scratch);
    }

    let nonzero_giant_rotations = schedule.giant_steps.iter().filter(|gs| gs.rot != 0).count();
    let has_nonzero_giant_rotation = nonzero_giant_rotations != 0;
    // Giant rotations rotate the post-product destination, so their keys are
    // the ones the destination's precision asks for. Every rotating step's key
    // is resolved once, indexed by step, because the steps need not share a
    // layout: the lazy route is only taken when all of them carry the
    // destination's base2k, and the accumulator is sized by the widest key
    // (each key-switch then clamps to its own width). An identity-only
    // transform legitimately resolves no key at all.
    let giant_keys: Vec<Option<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>> = schedule
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
        let batch = products.batch(module).min(giant_keys.len());
        let (mut prods, scratch_phase) = scratch.take_vec_znx_dft_slice_scratch(res.n().as_usize(), batch, cols, prod_size);
        let (mut lazy_acc_dft, mut scratch_phase) = scratch_phase.take_vec_znx_dft_scratch(res.n().as_usize(), cols, lazy_size);
        for col in 0..cols {
            module.vec_znx_dft_zero(&mut lazy_acc_dft, col);
        }

        let mut res_initialized = false;
        let mut prods: Vec<VecZnxDftBackendMut<'_, BE>> = prods.iter_mut().map(|prod| prod.to_backend_mut()).collect();
        for (g, giant_key) in giant_keys.iter().enumerate() {
            if g % batch == 0 {
                let steps = g..(g + batch).min(giant_keys.len());
                products.products(module, &mut prods[..steps.len()], steps, &mut scratch_phase);
            }
            let prod_dft = &mut prods[g % batch];
            let rot = schedule.giant_steps[g].rot;
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
                    crate::layouts::assert_no_row_offset("glwe_eval_linear_transformation", &key.key);
                    gglwe_product_from_dft(
                        module,
                        &mut lazy_acc_dft_backend,
                        prod_dft,
                        prod_base2k.as_usize(),
                        &key.key,
                        key_output_size,
                        nonzero_giant_rotations,
                        operand_galois_element(module, res.n().as_usize(), rot),
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
            products.products(
                module,
                core::slice::from_mut(&mut prod_dft_backend),
                g..g + 1,
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

/// The giant-step tree of [`crate::layouts::BsgsRotations::PowersOfTwo`].
struct GiantTree<'t> {
    /// The tree node of each giant step's product.
    leaves: &'t [usize],
    levels: &'t [GiantTreeLevel],
}

/// [`glwe_eval_giant_steps`] for [`crate::layouts::BsgsRotations::PowersOfTwo`]: every product
/// first, then the tree a level at a time, each level switching with one key.
///
/// The lazy route keeps the nodes in DFT, as the per-rotation one does its accumulator; the
/// fallback normalizes the products and runs the tree on ciphertexts. Both hold every
/// product at once, in buffers of their own: the scratch queries take no rotation count.
#[allow(clippy::type_complexity)]
fn glwe_eval_giant_tree<BE, R, H>(
    module: &Module<BE>,
    tree: &GiantTree<'_>,
    res: &mut R,
    prod_base2k: Base2K,
    products: &dyn GiantProducts<BE>,
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
    H: GetAutomorphismKey<BE>,
{
    let n = res.n().as_usize();
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k();
    let prod_size = products.size();
    let cnv_offset_lo = products.cnv_offset_lo();
    let steps = tree.leaves.len();
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

    let batch = products.batch(module).min(steps);
    let shape = GiantTreeShape {
        prod_size,
        res_size: res.size(),
        products_bytes: products.tmp_bytes(module, n),
    };

    if !lazy {
        // The products, then each as a normalized ciphertext, carved from the scratch beside the
        // work of the tree ([`giant_tree_tmp_bytes`]).
        let need = steps * giant_tree_fallback_buffer_bytes::<BE, _>(module, n, cols, &shape)
            + level_keys
                .iter()
                .flatten()
                .map(|key| giant_tree_switch_bytes::<BE, _, _>(module, res, key, false, 0, 0))
                .fold(giant_tree_work_bytes::<BE, _>(module, false, &shape, 0), usize::max);
        assert!(
            scratch.available() >= need,
            "glwe_eval_linear_transformation: scratch below its query"
        );
        let (mut prods, scratch_1) = scratch.borrow().take_vec_znx_dft_slice_scratch(n, steps, cols, prod_size);
        let (mut leaves, mut scratch_1) = scratch_1.take_glwe_slice_scratch(steps, res);
        {
            let mut views: Vec<VecZnxDftBackendMut<'_, BE>> = prods.iter_mut().map(|p| p.to_backend_mut()).collect();
            for start in (0..steps).step_by(batch) {
                let range = start..(start + batch).min(steps);
                products.products(module, &mut views[range.clone()], range, &mut scratch_1);
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
                    cnv_offset_lo,
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
                prod_size.saturating_sub(res.size()),
            )
        })
        .max()
        .unwrap_or(res.size());
    let node_size = key_output_size.max(prod_size);
    // A product is stored in a buffer as wide as an accumulator when the layout can view its
    // limbs apart; it is read over its own limbs either way.
    let wide = BE::DFT_LIMBS_CONTIGUOUS && node_size > prod_size;

    // The switches of every level, `(dst, src, accumulate, copy)` over buffers numbered
    // products first: a pair adds into its even node when that is as wide as an accumulator,
    // else into an accumulator of its own, the even node copied there first.
    let mut sizes = vec![if wide { node_size } else { prod_size }; steps];
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
    let mut held = vec![prod_size; steps];
    held.resize(sizes.len(), node_size);

    let node_bytes: usize = sizes
        .iter()
        .map(|&size| BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, size)))
        .sum();
    // Carved from the scratch beside the work of the tree ([`giant_tree_tmp_bytes`]).
    let need = node_bytes
        + plan
            .iter()
            .zip(&level_keys)
            .filter_map(|(switches, key)| {
                key.as_ref()
                    .map(|key| giant_tree_switch_bytes::<BE, _, _>(module, res, key, true, switches.len(), node_size))
            })
            .fold(giant_tree_work_bytes::<BE, _>(module, true, &shape, node_size), usize::max);
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
            let mut views: Vec<VecZnxDftBackendMut<'_, BE>> =
                full.iter_mut().map(|p| p.with_limb_range_mut(0, prod_size)).collect();
            for start in (0..steps).step_by(batch) {
                let range = start..(start + batch).min(steps);
                products.products(module, &mut views[range.clone()], range, &mut arena.borrow());
            }
        }
        // The limbs past the product of a buffer accumulated in place.
        for (leaf, view) in full.iter_mut().enumerate() {
            if zero_tail[leaf] {
                let mut tail = view.with_limb_range_mut(prod_size, node_size);
                for col in 0..cols {
                    module.vec_znx_dft_zero(&mut tail, col);
                }
            }
        }
    }

    // Every buffer a view of its own, taken out for the switches of a level and put back.
    let mut slots: Vec<Option<VecZnxDftBackendMut<'_, BE>>> =
        buffers.into_iter().map(|buffer| Some(buffer.into_inner())).collect();
    let widest = plan.iter().map(Vec::len).max().unwrap_or(0);
    let (mut dsts, mut srcs) = (Vec::with_capacity(widest), Vec::with_capacity(widest));
    let (mut source_sizes, mut accumulate) = (Vec::with_capacity(widest), Vec::with_capacity(widest));
    for ((level, switches_of_level), key) in tree.levels.iter().zip(&plan).zip(&level_keys) {
        let Some(key) = key else { continue };
        for &(dst, _, _, copy) in switches_of_level {
            if let Some(even) = copy {
                let mut node = slots[dst].take().expect("a node is held once");
                let even = slots[even].as_ref().expect("a node is held once");
                glwe_dft_copy_dft(module, &mut node, &vec_znx_dft_backend_ref_from_mut(even));
                slots[dst] = Some(node);
            }
        }
        // The destinations and sources of the level, in switch order, each source over the
        // limbs it holds.
        for &(dst, src, accumulates, _) in switches_of_level {
            dsts.push(slots[dst].take().expect("a node is switched into once per level"));
            srcs.push(slots[src].take().expect("a node is switched from once per level"));
            source_sizes.push(held[src]);
            accumulate.push(accumulates);
        }
        BE::gglwe_products_from_dft(
            module,
            operand_galois_element(module, n, level.rot),
            switches,
            &mut dsts,
            key_output_size.min(key.size()),
            &accumulate,
            &mut srcs,
            &source_sizes,
            prod_base2k.as_usize(),
            &key.key,
            &mut arena.borrow(),
        );
        for ((&(dst, src, _, _), node), source) in switches_of_level.iter().zip(dsts.drain(..)).zip(srcs.drain(..)) {
            slots[dst] = Some(node);
            slots[src] = Some(source);
            held[dst] = node_size;
        }
        source_sizes.clear();
        accumulate.clear();
    }
    // The root is read no more: one inverse transform and normalization per column, over the
    // limbs it holds.
    let res_k = res.k().as_usize();
    let mut res_backend = res.to_backend_mut();
    let mut root_node = slots[root].take().expect("the root is held");
    let mut root_view = root_node.with_limb_range_mut(0, held[root]);
    for col in 0..cols {
        module.vec_znx_idft_normalize(
            &mut res_backend.data,
            res_base2k.as_usize(),
            res_k,
            cnv_offset_lo,
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
/// upper bounds of them in [`giant_tree_tmp_bytes`].
struct GiantTreeShape {
    /// Limbs of a giant step's product.
    prod_size: usize,
    /// Limbs of the destination.
    res_size: usize,
    /// Bytes the products of a batch take beside their buffers ([`GiantProducts::tmp_bytes`]).
    products_bytes: usize,
}

/// The sizes the products of [`PlainProducts`] lay their scratch out from.
pub(super) struct PlainProductsShape {
    pub(super) cnv_offset_hi: usize,
    /// Limbs of a baby rotation.
    pub(super) baby_size: usize,
    /// Limbs of a diagonal.
    pub(super) diagonal_size: usize,
    /// Limbs of a giant step's product.
    pub(super) prod_size: usize,
    /// Terms of a sum, one for a single transform.
    pub(super) terms: usize,
    /// Giant steps whose products are computed together.
    pub(super) batch: usize,
}

/// Bytes the products of a batch of [`PlainProducts`] take beside their buffers, of degree `n` and `cols` columns.
pub(super) fn plain_products_tmp_bytes<BE, M>(module: &M, n: usize, cols: usize, shape: &PlainProductsShape) -> usize
where
    BE: Backend,
    M: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf,
{
    // A later term of a sum is computed into a bucket per step of a batch, a streamed diagonal prepared into a slot
    // and multiplied a diagonal at a time.
    let bucket = if shape.terms > 1 {
        shape.batch * BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, shape.prod_size))
    } else {
        0
    };
    bucket
        + BE::scratch_aligned(module.bytes_of_cnv_pvec_right(n, 1, shape.diagonal_size, PrepareHint::OneShot))
        + glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(
            module,
            shape.cnv_offset_hi,
            shape.baby_size,
            shape.diagonal_size,
        )
        .max(module.cnv_apply_dft_tmp_bytes(shape.cnv_offset_hi, shape.prod_size, shape.baby_size, shape.diagonal_size))
        .max(module.cnv_prepare_right_tmp_bytes(shape.diagonal_size, shape.diagonal_size))
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
fn giant_tree_work_bytes<BE, M>(module: &M, lazy: bool, shape: &GiantTreeShape, node_size: usize) -> usize
where
    BE: Backend,
    M: VecZnxIdftNormalizeTmpBytes + GLWENormalize<BE>,
{
    let prod = shape.products_bytes;
    let normalize = if lazy {
        module.vec_znx_idft_normalize_tmp_bytes(shape.res_size, node_size)
    } else {
        module
            .vec_znx_idft_normalize_tmp_bytes(shape.res_size, shape.prod_size)
            .max(module.glwe_normalize_tmp_bytes())
    };
    prod.max(normalize)
}

/// Bytes of the switches of a level of the giant-step tree with `key`: of its `switches` nodes
/// of `node_size` limbs together on the lazy route, of a ciphertext of the layout of `res` on
/// the other.
fn giant_tree_switch_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    key: &K,
    lazy: bool,
    switches: usize,
    node_size: usize,
) -> usize
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
        BE::gglwe_products_from_dft_tmp_bytes(module, switches, res.rank().as_usize(), node_size, key)
    } else {
        module.glwe_automorphism_tmp_bytes(res, res, key)
    }
}

/// Bytes the giant-step tree of a transform on [`crate::layouts::BsgsRotations::PowersOfTwo`]
/// with the schedule `plan` takes from the scratch, the larger of its two routes, for products of `prod_size` limbs
/// taking `products_bytes` beside their buffers: `res` is the destination and `key` the widest giant key. Zero for
/// [`crate::layouts::BsgsRotations::PerRotation`].
pub(super) fn giant_tree_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    res: &R,
    prod_size: usize,
    products_bytes: usize,
    plan: &LinearTransformationPlan,
    key: &K,
) -> usize
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl,
    Module<BE>: ModuleN
        + GLWEAutomorphism<BE>
        + GLWENormalize<BE>
        + VecZnxBigBytesOf
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxGadgetDecomposeTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let Some((_, levels)) = plan.rotations.giant_tree(plan.giant_steps.iter().copied()) else {
        return 0;
    };
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    let steps = plan.giant_steps.len();
    let switches: usize = levels.iter().map(|level| level.pairs.len()).sum();
    let shape = GiantTreeShape {
        prod_size,
        res_size: res.max_size(),
        products_bytes,
    };
    // A key-switch output is no wider than its key, and a tree without one keeps nodes as wide as the destination.
    let node_size = key.size().max(shape.prod_size).max(shape.res_size);
    // A buffer per giant step and at most one per switch, as wide as a node, beside the switches
    // of the level of most switches.
    let widest = levels.iter().map(|level| level.pairs.len()).max().unwrap_or(0);
    let lazy = (steps + switches) * BE::scratch_aligned(module.bytes_of_vec_znx_dft(n, cols, node_size))
        + giant_tree_work_bytes::<BE, _>(module, true, &shape, node_size)
            .max(giant_tree_switch_bytes::<BE, _, _>(module, res, key, true, widest, node_size));
    let fallback = steps * giant_tree_fallback_buffer_bytes::<BE, _>(module, n, cols, &shape)
        + giant_tree_work_bytes::<BE, _>(module, false, &shape, node_size)
            .max(giant_tree_switch_bytes::<BE, _, _>(module, res, key, false, 0, node_size));
    lazy.max(fallback)
}

/// The buckets [`sum_giant_prods`] computes the later terms of a sum into: one per product of a
/// batch of `batch` products of `size` limbs, and none for a single transform.
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

/// PROD of the giant steps `steps` summed over the terms `(babies[t], lts[t])`, `prods[i]` receiving step
/// `steps.start + i`, through [`DiagonalProd::accumulate_giant_prods`]: the first term writes the products, each
/// later one is computed into the first `prods.len()` of `buckets` ([`take_sum_buckets`]) and added in DFT.
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
