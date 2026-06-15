//! Reference implementations of the linear-transformation eval / baby-prep
//! methods.
//!
//! These free `*_default` functions are the reference algorithms a backend
//! forwards to from its [`crate::oep::LinearTransformationDefault`] impl (see
//! [`crate::impl_linear_transformation_defaults_full`]). The prepared path
//! follows docs/lt_bsgs.md §6: hoisted baby rotations, DFT-domain inner
//! products, lazy giant rotations, and one final BIG normalization.

#![allow(clippy::too_many_arguments)]
#![allow(private_bounds)]

use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, VecZnxAutomorphismAssignBackend, VecZnxBigAddAssign, VecZnxBigAddSmallAssign,
        VecZnxBigAlloc, VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf,
        VecZnxBigFromSmallBackend, VecZnxBigNormalize, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply,
        VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpA,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, GaloisElement, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain, LinearTransformation,
    default::{
        keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
        linear_transformation::{
            baby_steps::{glwe_prepare_linear_transformation_lhs, glwe_prepare_linear_transformation_lhs_tmp_bytes},
            inner_product::glwe_accumulate_prepared_baby_steps_dft_tmp_bytes,
            lazy::{glwe_lazy_giant_automorphism_from_dft_tmp_bytes, glwe_lazy_giant_automorphism_tmp_bytes},
            prepared_giants::{DiagonalProd, glwe_eval_giant_steps},
        },
    },
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, ModuleCoreAlloc,
        prepared::GGLWEPreparedToBackendRef,
    },
};

use super::LinearTransformationBabySteps;
use crate::layouts::prepared::PreparedDiagonal;

/// HAL/op bounds required by the eval reference path. Repeated on each free
/// function so backends only pull in what a method actually needs.
pub fn glwe_eval_linear_transformation_tmp_bytes_default<BE, M, R, A, B, K>(module: &M, res: &R, a: &A, pt: &B, key: &K) -> usize
where
    BE: Backend,
    M: poulpy_hal::api::ModuleN
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
        + VecZnxIdftApplyTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
{
    // The lazy prepared evaluator keeps PROD, giant rotations, and the
    // cross-giant accumulator in DFT; incompatible bases normalize through a
    // one-column BIG scratch before regular GLWE automorphism. Size both routes
    // and take the larger budget.
    let cols = a.rank().as_usize() + 1;
    let cnv_offset_hi = pt.size().saturating_sub(1);
    let prod_size = a.size() + pt.size() - cnv_offset_hi;
    let inner_dft = glwe_accumulate_prepared_baby_steps_dft_tmp_bytes::<BE, _>(module, cnv_offset_hi, a.size(), pt.size());
    let prod_col_big = module.bytes_of_vec_znx_big(1, prod_size);
    let prod_dft = module.bytes_of_vec_znx_dft(cols, prod_size);
    let lazy_size = key.size().max(prod_size);
    let lazy_acc_dft = module.bytes_of_vec_znx_dft(cols, lazy_size);
    let lazy_acc_big = module.bytes_of_vec_znx_big(cols, lazy_size);
    let rot_dft = module.bytes_of_vec_znx_dft(cols, key.size());
    let prepare_right = module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size());
    let lazy_dft =
        glwe_lazy_giant_automorphism_from_dft_tmp_bytes::<BE, _, _>(module, a.rank().as_usize(), prod_size, key, key.size());
    let fallback_path = prod_dft + prod_col_big + inner_dft;
    let lazy_dft_rot = rot_dft + lazy_dft;
    let lazy_dft_path = prod_dft + lazy_acc_dft + inner_dft + lazy_dft_rot + lazy_acc_big;

    module
        .glwe_automorphism_tmp_bytes(res, a, key)
        .max(module.glwe_mul_plain_tmp_bytes(res, a, pt))
        .max(prepare_right)
        .max(glwe_prepare_linear_transformation_lhs_tmp_bytes::<BE, _, _, _>(
            module, a, key,
        ))
        .max(glwe_lazy_giant_automorphism_tmp_bytes::<BE, _, _, _>(
            module,
            res,
            key,
            key.size(),
        ))
        .max(fallback_path)
        .max(lazy_dft_path)
}

/// Reference impl: scratch bytes for [`glwe_prepare_linear_transformation_lhs_default`].
///
/// Sizes both the hoisted baby route (DFT the mask once, VMP per key) and the
/// plain per-baby `glwe_automorphism` fallback, and takes the larger.
pub fn glwe_prepare_linear_transformation_lhs_tmp_bytes_default<BE, M, A, K>(module: &M, a: &A, key: &K) -> usize
where
    BE: Backend,
    M: poulpy_hal::api::ModuleN
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
{
    glwe_prepare_linear_transformation_lhs_tmp_bytes::<BE, _, _, _>(module, a, key)
}

/// Reference impl: Phase A — materialize the hoisted baby-step rotations.
///
/// Fills the pre-allocated `cache` with `rot(a, k)` (prepared as `CnvPVecL`) for
/// every baby rotation `k` it already holds, reusing one DFT of the input mask
/// across all keys (docs/lt_bsgs.md §6.2). The LHS is independent of the matrix
/// diagonals, so the same prepared cache is reused across every giant step and
/// across transforms that share the input. `a_effective_k` is the CKKS-supplied
/// base2k alignment for the input. Forwards to the internal
/// `glwe_prepare_linear_transformation_lhs`.
pub fn glwe_prepare_linear_transformation_lhs_default<BE, M, A, H, K>(
    module: &M,
    cache: &mut LinearTransformationBabySteps<BE>,
    a: &A,
    a_effective_k: usize,
    keys: &H,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecAlloc<BE>
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
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    glwe_prepare_linear_transformation_lhs(module, cache, a, a_effective_k, keys, key_size, scratch);
}

/// Reference impl: prepared BSGS evaluation of a linear transformation.
///
/// Evaluates `M·v` from the prepared left operand `lhs` (baby rotations, Phase
/// A) and the prepared right operand `rhs` (matrix diagonals), writing the
/// result into `res`. This is Phases B/C of docs/lt_bsgs.md §6: per-giant PROD,
/// lazy giant rotations, and one final normalization. `cnv_offset` is the
/// CKKS-supplied limb alignment between the input and diagonal scales. Both
/// operands must have been prepared for the same BSGS schedule.
///
/// Asserts at least one non-empty giant step (a fully-pruned transform is a
/// caller bug), then delegates to the shared `glwe_eval_giant_steps` loop.
pub fn glwe_eval_linear_transformation_into_default<BE, M, R, H, K>(
    module: &M,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
    keys: &H,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
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
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert!(
        rhs.giant_steps.iter().any(|gs| !gs.diagonals.is_empty()),
        "linear transformation has no non-empty giant steps"
    );

    glwe_eval_giant_steps(module, cnv_offset, res, lhs, rhs, keys, key_size, scratch);
}

/// Reference impl: streamed (unprepared-RHS) evaluation.
///
/// Same result as [`glwe_eval_linear_transformation_into_default`] but consumes
/// the unprepared `rhs` directly, preparing each diagonal on the fly instead of
/// from a materialized `LinearTransformation<PreparedDiagonal>`. Lower peak memory,
/// higher compute — for memory-bound backends (e.g. GPU).
pub fn glwe_eval_linear_transformation_unprepared_rhs_into_default<BE, M, R, P, H, K>(
    module: &M,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<P>,
    keys: &H,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEAutomorphism<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
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
    P: GLWEToBackendRef<BE> + GLWEInfos + DiagonalProd<BE>,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert!(
        rhs.giant_steps.iter().any(|gs| !gs.diagonals.is_empty()),
        "linear transformation has no non-empty giant steps"
    );

    glwe_eval_giant_steps(module, cnv_offset, res, lhs, rhs, keys, key_size, scratch);
}

/// Reference impl: scratch bytes for the streamed (unprepared-RHS) evaluation.
///
/// The streamed inner product additionally holds one resident `CnvPVecR`
/// diagonal slot and a `cnv_prepare_right` scratch on top of the prepared
/// evaluation budget.
pub fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default<BE, M, R, A, B, K>(
    module: &M,
    res: &R,
    a: &A,
    pt: &B,
    key: &K,
) -> usize
where
    BE: Backend,
    M: poulpy_hal::api::ModuleN
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
        + VecZnxIdftApplyTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    B: GLWEInfos,
    K: GGLWEInfos,
{
    glwe_eval_linear_transformation_tmp_bytes_default::<BE, _, _, _, _, _>(module, res, a, pt, key)
        + module.bytes_of_cnv_pvec_right(1, pt.size())
        + module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
}
