//! The `_assign` and self-caching evaluations compose the evaluation over a caller's baby cache with a copy and a
//! baby-step preparation.
use poulpy_core::layouts::BabyStepOperand;
use poulpy_core::{
    DiagonalProd, GLWEBytesOf, LinearTransformation, LinearTransformationBabySteps, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformationPlan},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSCopyOps, LtDiagonalMeta},
    ckks_ensure,
    layouts::ScratchArenaTakeCKKS,
    oep::CKKSLinearTransformationImpl,
};

pub(crate) fn ckks_eval_linear_transformation_assign_derived<BE, Dst, P, H, L: BabyStepOperand<BE>>(
    module: &Module<BE>,
    dst: &mut Dst,
    babies: &LinearTransformationBabySteps<BE, L>,
    lt: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSLinearTransformationImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: DiagonalProd<BE> + LtDiagonalMeta,
    H: GetAutomorphismKey<BE> + poulpy_core::layouts::GetTensorKey<BE>,
{
    // The evaluation checks its own query on what the working copy leaves.
    ckks_ensure!(
        scratch.available() >= BE::scratch_aligned(module.glwe_bytes_of_from_infos(&*dst)),
        "ckks_eval_linear_transformation_assign: scratch below its query"
    );
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        BE::ckks_eval_linear_transformation_into(module, &mut tmp, dst, babies, lt, keys, &mut scratch_local)?;
        // `ckks_copy` also moves the metadata the evaluation stamped, which a `glwe_copy` would leave stale.
        module.ckks_copy(dst, &tmp, &mut scratch_local)
    })
}

pub(crate) fn ckks_eval_linear_transformation_self_into_derived<BE, Dst, Src, P, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    lt: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSLinearTransformationImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: DiagonalProd<BE> + LtDiagonalMeta,
    H: GetAutomorphismKey<BE> + poulpy_core::layouts::GetTensorKey<BE>,
{
    // Only the baby cache is carved: an unprepared `lt` streams its diagonals inside the evaluation, and the nested
    // calls check their own queries on what the cache leaves.
    let plan = lt.index();
    ckks_ensure!(
        scratch.available() >= module.linear_transformation_baby_steps_bytes_of(&plan, src),
        "ckks_eval_linear_transformation_self_into: scratch below its baby-step cache"
    );
    let (mut babies, mut scratch) = scratch
        .borrow()
        .take_linear_transformation_baby_steps_scratch(module, &plan, src);
    BE::ckks_prepare_linear_transformation_baby_steps(module, &mut babies, src, keys, &mut scratch)?;
    BE::ckks_eval_linear_transformation_into(module, dst, src, &babies, lt, keys, &mut scratch)
}

pub(crate) fn ckks_eval_linear_transformation_self_assign_derived<BE, Dst, P, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    lt: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSLinearTransformationImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: DiagonalProd<BE> + LtDiagonalMeta,
    H: GetAutomorphismKey<BE> + poulpy_core::layouts::GetTensorKey<BE>,
{
    // The evaluation checks its own query on what the working copy leaves.
    ckks_ensure!(
        scratch.available() >= BE::scratch_aligned(module.glwe_bytes_of_from_infos(&*dst)),
        "ckks_eval_linear_transformation_self_assign: scratch below its query"
    );
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        BE::ckks_eval_linear_transformation_self_into(module, &mut tmp, dst, lt, keys, &mut scratch_local)?;
        module.ckks_copy(dst, &tmp, &mut scratch_local)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_eval_linear_transformation_self_into_tmp_bytes_derived<BE, C, D, K, T>(
    module: &Module<BE>,
    streamed: bool,
    ct: &C,
    diagonal: &D,
    plan: &LinearTransformationPlan,
    tsk: &T,
    key: &K,
) -> usize
where
    BE: CKKSLinearTransformationImpl,
    C: CKKSCtBounds,
    D: GLWEInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    let eval = if streamed {
        BE::ckks_eval_linear_transformation_streamed_tmp_bytes(module, ct, diagonal, plan, tsk, key)
    } else {
        BE::ckks_eval_linear_transformation_tmp_bytes(module, ct, diagonal, plan, tsk, key)
    };
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(ct))
        + module.linear_transformation_baby_steps_bytes_of(plan, ct)
        + BE::ckks_prepare_linear_transformation_baby_steps_tmp_bytes(module, ct, plan, key).max(eval)
}
