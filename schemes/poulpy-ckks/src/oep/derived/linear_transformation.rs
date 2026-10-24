//! The `_assign` and self-caching evaluations compose the evaluation over a caller's baby cache with a copy and a
//! baby-step preparation.
use poulpy_core::{
    DiagonalProd, LinearTransformation, LinearTransformationBabySteps,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSCopyOps, LtDiagonalMeta},
    layouts::ScratchArenaTakeCKKS,
    oep::CKKSLinearTransformationImpl,
};

pub(crate) fn ckks_eval_linear_transformation_assign_derived<BE, Dst, P, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    babies: &LinearTransformationBabySteps<BE>,
    lt: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSLinearTransformationImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
    H: GetAutomorphismKey<BE>,
{
    // The working copy is carved from scratch, which `ckks_eval_linear_transformation_tmp_bytes` covers.
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
    P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
    H: GetAutomorphismKey<BE>,
{
    // Only the baby cache is allocated: a plaintext `lt` streams its diagonals inside the evaluation.
    let mut babies = LinearTransformationBabySteps::alloc_for(module, lt, src);
    BE::ckks_prepare_linear_transformation_baby_steps(module, &mut babies, src, keys, scratch)?;
    BE::ckks_eval_linear_transformation_into(module, dst, src, &babies, lt, keys, scratch)
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
    P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
    H: GetAutomorphismKey<BE>,
{
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        BE::ckks_eval_linear_transformation_self_into(module, &mut tmp, dst, lt, keys, &mut scratch_local)?;
        module.ckks_copy(dst, &tmp, &mut scratch_local)
    })
}
