//! Reference implementations of the linear-transformation eval / baby-prep
//! methods.
//!
//! These free `*_reference` functions are the reference algorithms a backend
//! forwards to from its [`crate::oep::LinearTransformationImpl`] impl (see
//! [`crate::impl_linear_transformation_reference_full`]). The prepared path
//! follows docs/guides/linear_transformation.md: hoisted baby rotations, DFT-domain inner
//! products, lazy giant rotations, and one final BIG normalization.

#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::Module;
use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, VecZnxAutomorphismAssign, VecZnxBigAddAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, GaloisElement, PrepareHint, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation,
    gadget::prod_dft_size,
    layouts::{
        DiagonalProd, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformationPlan,
        ModuleCoreAlloc, assert_linear_transformation_into, assert_linear_transformation_sum,
    },
    reference::{
        keyswitching::{GGLWEProductReference, GLWEKeyswitchInternal},
        linear_transformation::{
            baby_steps::{
                glwe_prepare_baby_doubling_tmp_bytes, glwe_prepare_linear_transformation_baby_steps,
                glwe_prepare_linear_transformation_baby_steps_tmp_bytes,
            },
            inner_product::glwe_accumulate_prepared_baby_steps_dft_tmp_bytes,
            lazy::glwe_lazy_giant_automorphism_tmp_bytes,
            prepared_giants::{PlainProductsShape, giant_tree_tmp_bytes, glwe_eval_giant_steps, plain_products_tmp_bytes},
        },
    },
};

use super::{LinearTransformationBabySteps, LinearTransformationPrepared};
use crate::api::GLWEBytesOf;

/// HAL/op bounds required by the eval reference path. Repeated on each free
/// function so backends only pull in what a method actually needs.
pub fn glwe_eval_linear_transformation_tmp_bytes_reference<BE, R, A, B, K, T>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    pt: &B,
    plan: &LinearTransformationPlan,
    tsk: &T,
    key: &K,
) -> usize
where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + crate::GLWEPreparedTensoring<BE>
        + CnvPVecBytesOf
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + Convolution<BE>
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    // The lazy prepared evaluator keeps PROD, giant rotations, and the
    // cross-giant accumulator in DFT; incompatible bases normalize through a
    // one-column BIG scratch before regular GLWE automorphism. Size both routes
    // and take the larger budget.
    let cols = a.rank().as_usize() + 1;
    if pt.rank().as_usize() > 0 {
        // A product of each giant step's pairs into DFT, sized at the larger of the tensor key and `res`.
        let res_infos = crate::layouts::GLWELayout {
            k: res.max_k(),
            ..res.glwe_layout()
        };
        let products_bytes = super::ciphertext_diagonals::ciphertext_products_tmp_bytes(
            module,
            &res_infos,
            a.max_size(),
            pt.max_size(),
            a.max_k().max(pt.max_k()),
            tsk,
        );
        let prod_size = tsk.size().max(res.max_size());
        return (glwe_prepare_linear_transformation_baby_steps_tmp_bytes::<BE, _, _>(module, a, key)
            + glwe_prepare_baby_doubling_tmp_bytes::<BE, _, _, _>(module, a, plan, key))
        .max(giant_routes_tmp_bytes(module, cols, prod_size, products_bytes, 1, res, key))
        .max(giant_tree_tmp_bytes(module, res, prod_size, products_bytes, plan, key));
    }
    // Scratch is allocated up-front and must cover the physical working set,
    // so the budget is sized off the operands' allocated width (`max_size()`)
    // rather than their meta-derived `size()`.
    let a_size = a.max_size();
    let pt_size = pt.max_size();
    // The query has no conversion-offset argument: offset zero retains the
    // widest product, so reserve that working set for every supported offset.
    let min_cnv_offset = 0;
    let prod_size = a_size + pt_size;
    let inner_dft = glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(module, min_cnv_offset, a_size, pt_size);
    let prepare_right = module.cnv_prepare_right_tmp_bytes(pt_size, pt_size);
    let tree_products = PlainProductsShape {
        cnv_offset_hi: min_cnv_offset,
        baby_size: a_size,
        diagonal_size: pt_size,
        prod_size: prod_dft_size(a_size, pt_size, min_cnv_offset),
        terms: 1,
        batch: BE::glwe_giant_step_batch(module).min(plan.giant_steps.len()),
    };
    let tree_products_bytes =
        plain_products_tmp_bytes::<BE, _>(module, res.n().as_usize(), res.rank().as_usize() + 1, &tree_products);

    module
        .glwe_automorphism_tmp_bytes(res, a, key)
        .max(module.glwe_mul_plain_tmp_bytes(res, a, pt))
        .max(prepare_right)
        .max(
            glwe_prepare_linear_transformation_baby_steps_tmp_bytes::<BE, _, _>(module, a, key)
                + glwe_prepare_baby_doubling_tmp_bytes::<BE, _, _, _>(module, a, plan, key),
        )
        .max(glwe_lazy_giant_automorphism_tmp_bytes::<BE, _, _, _>(module, res, key))
        .max(giant_routes_tmp_bytes(
            module,
            cols,
            prod_size,
            inner_dft,
            BE::glwe_giant_step_batch(module),
            res,
            key,
        ))
        .max(giant_tree_tmp_bytes(
            module,
            res,
            tree_products.prod_size,
            tree_products_bytes,
            plan,
            key,
        ))
}

/// Bytes the per-rotation routes of the giant-step loop take for products of `prod_size` limbs and `cols` columns,
/// `batch` held at once, taking `products_bytes` beside them, into `res` with giant keys no wider than `key`: the lazy
/// route holds the products of a batch and the accumulator, the fallback first normalizes each product into the
/// destination layout, then rotates that accumulator in place (both operands therefore use `res`).
fn giant_routes_tmp_bytes<BE, R, K>(
    module: &Module<BE>,
    cols: usize,
    prod_size: usize,
    products_bytes: usize,
    batch: usize,
    res: &R,
    key: &K,
) -> usize
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl,
    Module<BE>: poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GGLWEProductReference<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + VecZnxIdftNormalizeTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let prod_col_big = module.bytes_of_vec_znx_big(module.n(), 1, prod_size);
    let prod_dft = module.bytes_of_vec_znx_dft(module.n(), cols, prod_size);
    let lazy_size = key.size().max(prod_size);
    let lazy_acc_dft = module.bytes_of_vec_znx_dft(module.n(), cols, lazy_size);
    let lazy_acc_big = module.bytes_of_vec_znx_big(module.n(), cols, lazy_size);
    let lazy_dft = crate::reference::keyswitching::glwe::gglwe_product_from_dft_tmp_bytes(module, cols - 1, prod_size, key);
    let fallback_work = products_bytes
        .max(module.glwe_automorphism_tmp_bytes(res, res, key))
        .max(module.vec_znx_normalize_tmp_bytes())
        .max(module.vec_znx_big_normalize_tmp_bytes())
        .max(module.vec_znx_idft_apply_tmp_bytes());
    let fallback_path = BE::scratch_aligned(prod_dft) + BE::scratch_aligned(prod_col_big) + fallback_work;
    let lazy_final = BE::scratch_aligned(lazy_acc_big)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lazy_dft_path =
        batch * BE::scratch_aligned(prod_dft) + BE::scratch_aligned(lazy_acc_dft) + products_bytes.max(lazy_dft).max(lazy_final);
    fallback_path.max(lazy_dft_path)
}

/// Reference impl: scratch bytes for [`glwe_prepare_linear_transformation_baby_steps_reference`].
///
/// Sizes both the hoisted baby route (DFT the mask once, VMP per key) and the
/// plain per-baby `glwe_automorphism` fallback, and takes the larger.
pub fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference<BE, A, K>(
    module: &Module<BE>,
    a: &A,
    plan: &LinearTransformationPlan,
    key: &K,
) -> usize
where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + crate::api::GLWENormalize<BE>,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    glwe_prepare_linear_transformation_baby_steps_tmp_bytes::<BE, _, _>(module, a, key)
        + glwe_prepare_baby_doubling_tmp_bytes::<BE, _, _, _>(module, a, plan, key)
}

/// Reference impl: Phase A — materialize the hoisted baby-step rotations.
///
/// Fills the pre-allocated `cache` with `rot(a, k)` (prepared as `CnvPVecL`) for
/// every baby rotation `k` it already holds, reusing one DFT of the input mask
/// across all keys (docs/guides/linear_transformation.md). The LHS is independent of the matrix
/// diagonals, so the same prepared cache is reused across every giant step and
/// across transforms that share the input. Forwards to the internal
/// `glwe_prepare_linear_transformation_baby_steps`.
pub fn glwe_prepare_linear_transformation_baby_steps_reference<BE, A, H>(
    module: &Module<BE>,
    cache: &mut LinearTransformationBabySteps<BE>,
    a: &A,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + crate::oep::GGLWEProductDigitsStridedImpl + crate::oep::LinearTransformationImpl,
    Module<BE>: GLWEBytesOf<BE>
        + CnvPVecAlloc<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + poulpy_hal::api::ModuleN
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + GaloisElement
        + crate::api::GLWENormalize<BE>
        + Sync
        + poulpy_hal::api::VecZnxAutomorphism<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
{
    glwe_prepare_linear_transformation_baby_steps(module, cache, a, keys, scratch);
    cache.set_k(a.k());
}

/// Reference impl: BSGS evaluation of a linear transformation, generic over the
/// diagonal representation `P`.
///
/// Evaluates `M·v` from the prepared left operand `lhs` (baby rotations, Phase A)
/// and the right operand `rhs` (matrix diagonals), writing the result into `res`.
/// This is Phases B/C of docs/guides/linear_transformation.md: per-giant PROD, lazy giant rotations,
/// and one final normalization. `cnv_offset` is the CKKS-supplied limb alignment
/// between the input and diagonal scales. The per-giant PROD is dispatched by the
/// [`DiagonalProd::operand`] of `P`, so `P = PreparedDiagonal` runs the resident fused path
/// and an unprepared `P` streams each diagonal; the rest of the loop is shared.
///
/// Runs [`assert_linear_transformation_into`] (a fully-pruned transform is a
/// caller bug), then the shared `glwe_eval_giant_steps` loop.
pub fn glwe_eval_linear_transformation_into_reference<BE, R, P, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + crate::GLWEPreparedTensoring<BE>
        + crate::reference::operations::GLWETensoringReference<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + poulpy_hal::api::ModuleN
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
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
        + VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxDftAlloc<BE>
        + VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEMulPlain<BE>
        + GaloisElement
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    H: GetAutomorphismKey<BE> + crate::layouts::GetTensorKey<BE>,
{
    assert_linear_transformation_into(lhs, rhs);
    glwe_eval_giant_steps(module, cnv_offset, res, &[lhs], &[rhs], keys, scratch);
    res.set_canonical(true);
}

/// Reference impl: scratch bytes for [`glwe_eval_linear_transformation_sum_into_reference`],
/// the single-transform budget plus the PROD bucket every term after the first is computed into.
pub fn glwe_eval_linear_transformation_sum_tmp_bytes_reference<BE, R, A, B, K, T>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    pt: &B,
    plan: &LinearTransformationPlan,
    tsk: &T,
    key: &K,
) -> usize
where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + crate::GLWEPreparedTensoring<BE>
        + CnvPVecBytesOf
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + Convolution<BE>
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    // A batch of buckets, each sized like the single budget's PROD buffer.
    let bucket = BE::glwe_giant_step_batch(module)
        * BE::scratch_aligned(module.bytes_of_vec_znx_dft(module.n(), a.rank().as_usize() + 1, a.max_size() + pt.max_size()));
    glwe_eval_linear_transformation_tmp_bytes_reference::<BE, _, _, _, _, _>(module, res, a, pt, plan, tsk, key) + bucket
}

/// Reference impl: `res = Σ_t M_t(a_t)` for resident transforms sharing one giant schedule.
///
/// `babies[t]` is the baby cache of `a_t` and `lts[t]` is `M_t`. Each giant step's PROD is
/// summed over the terms in DFT, then rotated once by the
/// shared giant loop, so one term is [`glwe_eval_linear_transformation_into_reference`].
///
/// Runs [`assert_linear_transformation_sum`] first: every term shares the first
/// one's giant schedule, first diagonal's layout, width and scale, and source
/// layout, so one schedule, PROD bucket and `cnv_offset` serve them all.
pub fn glwe_eval_linear_transformation_sum_into_reference<BE, R, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformationPrepared<BE>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + crate::GLWEPreparedTensoring<BE>
        + crate::reference::operations::GLWETensoringReference<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>
        + GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + poulpy_hal::api::ModuleN
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
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
        + VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxDftAlloc<BE>
        + VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEMulPlain<BE>
        + GaloisElement
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE> + crate::layouts::GetTensorKey<BE>,
{
    assert_linear_transformation_sum(babies, lts);
    glwe_eval_giant_steps(module, cnv_offset, res, babies, lts, keys, scratch);
    res.set_canonical(true);
}

/// Reference impl: scratch bytes for the streamed (unprepared-RHS) evaluation.
///
/// The streamed inner product additionally holds one resident `CnvPVecR`
/// diagonal slot and a `cnv_prepare_right` scratch on top of the prepared
/// evaluation budget, both sized at the module degree, an upper bound for a
/// diagonal of any degree.
pub fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference<BE, R, A, B, K, T>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    pt: &B,
    plan: &LinearTransformationPlan,
    tsk: &T,
    key: &K,
) -> usize
where
    BE: Backend
        + crate::oep::GGLWEProductDigitsStridedImpl
        + crate::oep::LinearTransformationImpl
        + crate::oep::GLWETensoringImpl,
    Module<BE>: GLWEBytesOf<BE>
        + crate::GLWEPreparedTensoring<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + CnvPVecBytesOf
        + Convolution<BE>
        + GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    glwe_eval_linear_transformation_tmp_bytes_reference::<BE, _, _, _, _, _>(module, res, a, pt, plan, tsk, key)
        + module.bytes_of_cnv_pvec_right(module.n(), 1, pt.size(), PrepareHint::OneShot)
        + module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
}
