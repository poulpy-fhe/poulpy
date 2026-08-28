//! Reference implementations of the linear-transformation eval / baby-prep
//! methods.
//!
//! These free `*_default` functions are the reference algorithms a backend
//! forwards to from its [`crate::oep::LinearTransformationDefault`] impl (see
//! [`crate::impl_linear_transformation_defaults_full`]). The prepared path
//! follows docs/linear_transformation.md: hoisted baby rotations, DFT-domain inner
//! products, lazy giant rotations, and one final BIG normalization.

#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, VecZnxAutomorphismAssignBackend, VecZnxAutomorphismAssignTmpBytes,
        VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigAutomorphismAssign,
        VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmallBackend, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, GaloisElement, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation,
    default::{
        automorphism::glwe::glwe_automorphism_tmp_bytes_upper_default,
        keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal, bound_accumulation_output_size_with_tail},
        linear_transformation::{
            baby_steps::{
                glwe_prepare_linear_transformation_baby_steps, glwe_prepare_linear_transformation_baby_steps_bound_tmp_bytes,
                glwe_prepare_linear_transformation_baby_steps_tmp_bytes,
            },
            inner_product::glwe_accumulate_prepared_baby_steps_dft_tmp_bytes,
            lazy::{glwe_lazy_giant_automorphism_from_dft_bound_tmp_bytes, glwe_lazy_giant_automorphism_from_dft_tmp_bytes},
            prepared_giants::{DiagonalProd, glwe_eval_giant_steps},
        },
        operations::GLWENormalizeDefault,
    },
    layouts::{
        GGLWEInfos, GGLWEUse, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWELayout,
        GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos, ModuleCoreAlloc, WithEffectiveDsize,
        prepared::GGLWEPreparedToBackendRef,
    },
};

use super::LinearTransformationBabySteps;
use crate::api::GLWEBytesOf;

/// HAL/op bounds required by the eval reference path. Repeated on each free
/// function so backends only pull in what a method actually needs.
pub fn glwe_eval_linear_transformation_tmp_bytes_default<BE, M, R, A, P, H, K>(
    module: &M,
    res: &R,
    a: &A,
    rhs: &LinearTransformation<P>,
    keys: &H,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + Convolution<BE>
        + GGLWEProductDefault<BE>
        + crate::default::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    A: GLWEInfos,
    P: LWEInfos,
    K: GGLWEInfos,
    H: GLWEAutomorphismKeyLayoutHelper<K>,
{
    // The lazy prepared evaluator keeps PROD, giant rotations, and the
    // cross-giant accumulator in DFT; incompatible bases normalize through a
    // one-column BIG scratch before regular GLWE automorphism. Size both routes
    // and take the larger.
    //
    // Every key-dependent term is taken over the rotations this transform
    // really visits, each resolved through the same helper at the same
    // precision the evaluation will resolve them at. A transform is the unit
    // that decides which keys are consulted, so it is the unit the query takes.
    let cols = a.rank().as_usize() + 1;
    // Scratch is allocated up-front and must cover the physical working set,
    // so the budget is sized off the operands' allocated width (`max_size()`)
    // rather than their meta-derived `size()`.
    let a_size = a.max_size();
    let pt_layout: GLWELayout = match rhs.first_diagonal_plaintext() {
        Some(pt) => GLWELayout {
            n: pt.n(),
            base2k: pt.base2k(),
            k: pt.k(),
            rank: a.rank(),
        },
        // A transform with no encoded diagonal evaluates nothing; the operand
        // itself bounds every convolution width from above.
        None => a.glwe_layout(),
    };
    let pt_size = pt_layout.max_size();
    let cnv_offset_hi = pt_size.saturating_sub(1);
    let prod_size = a_size + pt_size - cnv_offset_hi;
    let inner_dft = glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(module, cnv_offset_hi, a_size, pt_size);
    let prod_col_big = module.bytes_of_vec_znx_big(1, prod_size);
    let prod_dft = module.bytes_of_vec_znx_dft(cols, prod_size);
    let prepare_right = module.cnv_prepare_right_tmp_bytes(pt_size, pt_size);
    let fallback_path = prod_dft + prod_col_big + inner_dft;

    // A transform with no giant rotation still runs the lazy accumulator, sized
    // off the destination rather than off any key, so that path is part of the
    // key-independent budget.
    let keyless_lazy_size = res.max_size().max(prod_size);
    let keyless_lazy_path = prod_dft
        + module.bytes_of_vec_znx_dft(cols, keyless_lazy_size)
        + inner_dft
        + module.bytes_of_vec_znx_dft(cols, res.max_size())
        + module.bytes_of_vec_znx_big(cols, keyless_lazy_size)
        + module.vec_znx_idft_apply_tmp_bytes();

    let mut worst = module
        .glwe_mul_plain_tmp_bytes(res, a, &pt_layout)
        .max(prepare_right)
        .max(fallback_path)
        .max(keyless_lazy_path)
        .max(glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default::<
            BE,
            _,
            _,
            _,
            _,
        >(module, a, rhs.baby_steps(), keys));

    let nonzero_giant_rotations = lt_giant_key_rotations(rhs).count();
    let giant_plans = lt_giant_key_rotations(rhs)
        .map(|rot| {
            let (layout, effective_dsize) = keys
                .get_automorphism_key_layout_for(module.galois_element(rot), res.k())
                .unwrap_or_else(|e| panic!("linear-transformation giant rotation {rot}: {e}"));
            let key = layout.with_dsize(effective_dsize);
            let use_: GGLWEUse = crate::default::keyswitching::glwe::bound_for(&key, res.k());
            let key_output_size = bound_accumulation_output_size_with_tail::<BE, _>(
                res,
                &use_,
                nonzero_giant_rotations,
                prod_size.saturating_sub(res.size()),
            );
            (layout, effective_dsize, use_, key_output_size)
        })
        .collect::<Vec<_>>();
    let lazy_acc_size = giant_plans
        .iter()
        .map(|(_, _, _, output_size)| *output_size)
        .max()
        .unwrap_or_else(|| res.max_size())
        .max(prod_size);
    for (layout, effective_dsize, use_, key_output_size) in giant_plans {
        let key = layout.with_dsize(effective_dsize);
        worst = worst.max(lt_eval_tmp_bytes_for_key::<BE, _, _, _>(
            module,
            res,
            prod_size,
            inner_dft,
            prod_dft,
            prod_col_big,
            key_output_size,
            lazy_acc_size,
            &use_,
            &key,
        ));
    }
    worst
}

/// The key-dependent half of the eval budget, for one key.
///
/// Used only by exact per-factor queries. Proxy/whole-chain bounds deliberately
/// use [`lt_eval_upper_tmp_bytes_for_key`] instead.
fn lt_eval_tmp_bytes_for_key<BE, M, R, K>(
    module: &M,
    res: &R,
    prod_size: usize,
    inner_dft: usize,
    prod_dft: usize,
    prod_col_big: usize,
    key_output_size: usize,
    lazy_acc_size: usize,
    use_: &GGLWEUse,
    key: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + crate::default::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = res.rank().as_usize() + 1;
    let rotation = module.bytes_of_vec_znx_dft(cols, key_output_size)
        + glwe_lazy_giant_automorphism_from_dft_tmp_bytes::<BE, _>(
            module,
            res.rank().as_usize(),
            prod_size,
            key_output_size,
            use_,
        );
    let lazy_dft_path = prod_dft
        + module.bytes_of_vec_znx_dft(cols, lazy_acc_size)
        + inner_dft
            .max(rotation)
            .max(module.bytes_of_vec_znx_big(cols, lazy_acc_size) + module.vec_znx_idft_apply_tmp_bytes());
    let fallback = prod_dft + prod_col_big + inner_dft.max(module.glwe_automorphism_tmp_bytes(res, res, key));
    lazy_dft_path.max(fallback)
}

/// Conservative representative-key counterpart of
/// [`lt_eval_tmp_bytes_for_key`]. The outer lifetime nesting is unchanged; only
/// the lazy gadget product and regular-automorphism fallback are replaced by
/// their upper variants.
fn lt_eval_upper_tmp_bytes_for_key<BE, M, R, K>(
    module: &M,
    res: &R,
    prod_size: usize,
    inner_dft: usize,
    prod_dft: usize,
    prod_col_big: usize,
    key_output_size: usize,
    lazy_acc_size: usize,
    use_: &GGLWEUse,
    key: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GGLWEProductDefault<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = res.rank().as_usize() + 1;
    let rotation = module.bytes_of_vec_znx_dft(cols, key_output_size)
        + glwe_lazy_giant_automorphism_from_dft_bound_tmp_bytes::<BE, _>(
            module,
            res.rank().as_usize(),
            prod_size,
            key_output_size,
            use_,
        );
    let lazy_dft_path = prod_dft
        + module.bytes_of_vec_znx_dft(cols, lazy_acc_size)
        + inner_dft
            .max(rotation)
            .max(module.bytes_of_vec_znx_big(cols, lazy_acc_size) + module.vec_znx_idft_apply_tmp_bytes());
    let fallback_automorphism = glwe_automorphism_tmp_bytes_upper_default::<BE, _, _, _, _>(module, res, res, key);
    let fallback = prod_dft + prod_col_big + inner_dft.max(fallback_automorphism);
    lazy_dft_path.max(fallback)
}

/// Upper bound of [`glwe_eval_linear_transformation_tmp_bytes_default`] over any
/// transform whose diagonals are no wider than `pt` and whose rotations all
/// resolve to `key`.
///
/// For a caller that allocates one arena before it knows which transforms will
/// run through it, which is what the whole-pipeline queries do. A caller that
/// has the transform in hand should take the exact query instead: this one
/// cannot see which keys the transform's rotations resolve to, so it is only as
/// good as the representative key it is given.
pub fn glwe_eval_linear_transformation_bound_tmp_bytes_default<BE, M, R, A, B, K>(
    module: &M,
    res: &R,
    a: &A,
    pt: &B,
    key: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + Convolution<BE>
        + GGLWEProductDefault<BE>
        + crate::default::keyswitching::GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a.rank().as_usize() + 1;
    let a_size = a.max_size();
    let pt_size = pt.max_size();
    let cnv_offset_hi = pt_size.saturating_sub(1);
    let prod_size = a_size + pt_size - cnv_offset_hi;
    let inner_dft = glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(module, cnv_offset_hi, a_size, pt_size);
    let prod_dft = module.bytes_of_vec_znx_dft(cols, prod_size);
    let prod_col_big = module.bytes_of_vec_znx_big(1, prod_size);
    let fallback_path = prod_dft + prod_col_big + inner_dft;
    let use_: GGLWEUse = crate::default::keyswitching::glwe::bound_for(key, res.k());
    // This query has no concrete transform, so use the maximum possible number
    // of nonzero giant buckets for the ring as a conservative accumulation
    // bound. Exact transform queries use their actual count above.
    let key_output_size =
        bound_accumulation_output_size_with_tail::<BE, _>(res, &use_, module.n(), prod_size.saturating_sub(res.size()));

    module
        .glwe_mul_plain_tmp_bytes(res, a, pt)
        .max(module.cnv_prepare_right_tmp_bytes(pt_size, pt_size))
        .max(fallback_path)
        .max(glwe_prepare_linear_transformation_baby_steps_bound_tmp_bytes::<BE, _, _, _>(
            module, a, key,
        ))
        .max(lt_eval_upper_tmp_bytes_for_key::<BE, _, _, _>(
            module,
            res,
            prod_size,
            inner_dft,
            prod_dft,
            prod_col_big,
            key_output_size,
            key_output_size.max(prod_size),
            &use_,
            key,
        ))
}

/// [`glwe_eval_linear_transformation_bound_tmp_bytes_default`] for a streamed RHS.
pub fn glwe_eval_linear_transformation_unprepared_rhs_bound_tmp_bytes_default<BE, M, R, A, B, K>(
    module: &M,
    res: &R,
    a: &A,
    pt: &B,
    key: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + CnvPVecBytesOf
        + Convolution<BE>
        + GGLWEProductDefault<BE>
        + crate::default::keyswitching::GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
{
    glwe_eval_linear_transformation_bound_tmp_bytes_default::<BE, _, _, _, _, _>(module, res, a, pt, key)
        + module.bytes_of_cnv_pvec_right(1, pt.max_size())
        + module.cnv_prepare_right_tmp_bytes(pt.max_size(), pt.max_size())
}

/// Non-identity giant rotations the transform evaluates. Baby rotations are
/// sized separately at the source precision; these bind at the destination
/// precision because they rotate the post-PROD value.
pub(crate) fn lt_giant_key_rotations<P>(rhs: &LinearTransformation<P>) -> impl Iterator<Item = i64> + '_ {
    rhs.giant_steps
        .iter()
        .filter(|gs| !gs.diagonals.is_empty())
        .map(|gs| gs.rot)
        .filter(|&rot| rot != 0)
}

/// Reference impl: scratch bytes for [`glwe_prepare_linear_transformation_baby_steps_default`].
///
/// Sizes both the hoisted baby route (DFT the mask once, VMP per key) and the
/// plain per-baby `glwe_automorphism` fallback, and takes the larger.
pub fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default<BE, M, A, H, K>(
    module: &M,
    a: &A,
    rotations: &[i64],
    keys: &H,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GaloisElement
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes,
    A: GLWEInfos,
    K: GGLWEInfos,
    H: GLWEAutomorphismKeyLayoutHelper<K>,
{
    // The cache holds exactly these rotations, and the preparation resolves each
    // through the helper; so does this.
    let mut worst: usize = module.cnv_prepare_left_tmp_bytes(a.size(), a.size());
    for rot in rotations.iter().copied().filter(|&rot| rot != 0) {
        let (layout, effective_dsize) = keys
            .get_automorphism_key_layout_for(module.galois_element(rot), a.k())
            .unwrap_or_else(|e| panic!("baby-step rotation {rot}: {e}"));
        worst = worst.max(glwe_prepare_linear_transformation_baby_steps_tmp_bytes::<BE, _, _, _>(
            module,
            a,
            &layout.with_dsize(effective_dsize),
        ));
    }
    worst
}

/// Reference impl: Phase A — materialize the hoisted baby-step rotations.
///
/// Fills the pre-allocated `cache` with `rot(a, k)` (prepared as `CnvPVecL`) for
/// every baby rotation `k` it already holds, reusing one DFT of the input mask
/// across all keys (docs/linear_transformation.md). The LHS is independent of the matrix
/// diagonals, so the same prepared cache is reused across every giant step and
/// across transforms that share the input. `a_k` is the CKKS-supplied
/// base2k alignment for the input. Forwards to the internal
/// `glwe_prepare_linear_transformation_baby_steps`.
pub fn glwe_prepare_linear_transformation_baby_steps_default<BE, M, A, H, K>(
    module: &M,
    cache: &mut LinearTransformationBabySteps<BE>,
    a: &A,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + poulpy_hal::api::ModuleN
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + GaloisElement,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
{
    glwe_prepare_linear_transformation_baby_steps(module, cache, a, keys, scratch);
}

/// Reference impl: BSGS evaluation of a linear transformation, generic over the
/// diagonal representation `P`.
///
/// Evaluates `M·v` from the prepared left operand `lhs` (baby rotations, Phase A)
/// and the right operand `rhs` (matrix diagonals), writing the result into `res`.
/// This is Phases B/C of docs/linear_transformation.md: per-giant PROD, lazy giant rotations,
/// and one final normalization. `cnv_offset` is the CKKS-supplied limb alignment
/// between the input and diagonal scales. The per-giant PROD is dispatched by `P`
/// via [`DiagonalProd`], so `P = PreparedDiagonal` runs the resident fused path
/// and a plaintext `P` streams each diagonal — the rest of the loop is shared.
///
/// Asserts at least one non-empty giant step (a fully-pruned transform is a
/// caller bug), then delegates to the shared `glwe_eval_giant_steps` loop.
pub fn glwe_eval_linear_transformation_into_default<BE, M, R, P, H, K>(
    module: &M,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CnvPVecBytesOf
        + Convolution<BE>
        + poulpy_hal::api::ModuleN
        + GGLWEProductDefault<BE>
        + GLWEKeyswitchInternal<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxIdftApplyTmpBytes
        + GLWEMulPlain<BE>
        + GaloisElement,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
{
    assert!(
        rhs.giant_steps.iter().any(|gs| !gs.diagonals.is_empty()),
        "linear transformation has no non-empty giant steps"
    );

    glwe_eval_giant_steps(module, cnv_offset, res, lhs, rhs, keys, scratch);
}

/// Reference impl: scratch bytes for the streamed (unprepared-RHS) evaluation.
///
/// The streamed inner product additionally holds one resident `CnvPVecR`
/// diagonal slot and a `cnv_prepare_right` scratch on top of the prepared
/// evaluation budget.
pub fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default<BE, M, R, A, P, H, K>(
    module: &M,
    res: &R,
    a: &A,
    rhs: &LinearTransformation<P>,
    keys: &H,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + CnvPVecBytesOf
        + Convolution<BE>
        + GGLWEProductDefault<BE>
        + crate::default::keyswitching::GLWEKeyswitchInternal<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VmpPMatBytesOf,
    R: GLWEInfos,
    A: GLWEInfos,
    P: LWEInfos,
    K: GGLWEInfos,
    H: GLWEAutomorphismKeyLayoutHelper<K>,
{
    let pt_size = rhs.first_diagonal_plaintext().map_or(a.max_size(), |pt| pt.max_size());
    glwe_eval_linear_transformation_tmp_bytes_default::<BE, _, _, _, _, _, _>(module, res, a, rhs, keys)
        + module.bytes_of_cnv_pvec_right(1, pt_size)
        + module.cnv_prepare_right_tmp_bytes(pt_size, pt_size)
}
