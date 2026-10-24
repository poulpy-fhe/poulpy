//! Sums, multiply-accumulate and affine forms compose the selected products and additions.
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps},
    ckks_ensure,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKS},
    oep::{CKKSAddManyImpl, CKKSAffineImpl, CKKSMulAddImpl, CKKSMulSubImpl},
};

/// Carves a `dst`-shaped temporary in a scratch scope, runs the product into it,
/// then folds it into `dst`.
fn mul_then_combine<BE, Dst, MulF, CombineF>(
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
    mul: MulF,
    combine: CombineF,
) -> Result<()>
where
    BE: Backend,
    Dst: GLWEInfos + CKKSInfos,
    MulF: for<'t> FnOnce(&mut CKKSCiphertextViewMut<'t, BE>, &mut ScratchArena<'t, BE>) -> Result<()>,
    CombineF: for<'t> FnOnce(&mut Dst, &CKKSCiphertextViewMut<'t, BE>, &mut ScratchArena<'t, BE>) -> Result<()>,
{
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        mul(&mut tmp, &mut scratch_local)?;
        combine(dst, &tmp, &mut scratch_local)
    })
}

pub(crate) fn ckks_add_many_tmp_bytes_derived<BE, R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
where
    BE: CKKSAddManyImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
{
    module
        .ckks_add_tmp_bytes(res.size())
        .max(module.glwe_normalize_tmp_bytes())
        .max(module.ckks_copy_tmp_bytes(res, a))
}

pub(crate) fn ckks_add_many_derived<BE, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &[&A],
    normalize_after: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAddManyImpl,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    const OP: &str = "ckks_add_many";
    let invalid = |reason: &str| -> Result<()> {
        Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: reason.into(),
        }
        .into())
    };
    if normalize_after == 0 {
        return invalid("normalize_after must be at least 1");
    }
    let [first, rest @ ..] = a else {
        return invalid("a must contain at least one ciphertext");
    };
    ckks_ensure!(
        scratch.available() >= ckks_add_many_tmp_bytes_derived(module, &*res, *first),
        "{OP}: scratch below its query"
    );
    match rest {
        [] => module.ckks_copy(res, *first, scratch)?,
        [second, rest @ ..] => {
            module.ckks_add_into(res, *first, *second, scratch)?;
            for (additions, term) in (1..).zip(rest) {
                if additions % normalize_after == 0 {
                    module.glwe_normalize_assign(res, scratch);
                }
                module.ckks_add_assign(res, *term, scratch)?;
            }
        }
    }
    res.set_noise(None);
    Ok(())
}

pub(crate) fn ckks_mul_add_ct_tmp_bytes_derived<BE, R, A, B, T>(module: &Module<BE>, res: &R, a: &A, b: &B, tsk: &T) -> usize
where
    BE: CKKSMulAddImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    T: GGLWEInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_tmp_bytes(res, a, b, tsk)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_add_pt_vec_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulAddImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_vec_tmp_bytes(res, a, b)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_add_pt_const_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulAddImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_const_tmp_bytes(res, a, b)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_add_ct_into_derived<BE, Dst, A, B, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    b: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulAddImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetTensorKey<BE>,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_into(tmp, a, b, tsk, s),
        |dst, tmp, s| module.ckks_add_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_add_pt_vec_into_derived<BE, Dst, A, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulAddImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_vec_into(tmp, a, pt, s),
        |dst, tmp, s| module.ckks_add_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_add_pt_const_into_derived<BE, Dst, A, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    pt_coeff: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulAddImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
        |dst, tmp, s| module.ckks_add_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_affine_pt_const_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, affine_const: &P) -> usize
where
    BE: CKKSAffineImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module
        .ckks_mul_pt_const_tmp_bytes(res, a, affine_const)
        .max(module.ckks_add_pt_const_tmp_bytes(res.size()))
}

pub(crate) fn ckks_affine_pt_const_into_derived<BE, Dst, A, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    affine_const: &P,
    offset_coeff: usize,
    scale_coeff: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    module.ckks_mul_pt_const_into(dst, a, affine_const, scale_coeff, scratch)?;
    module.ckks_add_pt_const_assign(dst, 0, affine_const, offset_coeff, scratch)
}

pub(crate) fn ckks_affine_pt_const_assign_derived<BE, Dst, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    affine_const: &P,
    offset_coeff: usize,
    scale_coeff: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    module.ckks_mul_pt_const_assign(dst, affine_const, scale_coeff, scratch)?;
    module.ckks_add_pt_const_assign(dst, 0, affine_const, offset_coeff, scratch)
}

pub(crate) fn ckks_affine_pt_vec_tmp_bytes_derived<BE, R, A, S>(module: &Module<BE>, res: &R, a: &A, scale: &S) -> usize
where
    BE: CKKSAffineImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    S: CKKSInfos,
{
    module
        .ckks_mul_pt_vec_tmp_bytes(res, a, scale)
        .max(module.ckks_add_pt_vec_tmp_bytes(res.size()))
}

pub(crate) fn ckks_affine_pt_vec_into_derived<BE, Dst, A, S, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    scale: &S,
    offset: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    module.ckks_mul_pt_vec_into(dst, a, scale, scratch)?;
    module.ckks_add_pt_vec_assign(dst, offset, scratch)
}

pub(crate) fn ckks_affine_pt_vec_assign_derived<BE, Dst, S, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    scale: &S,
    offset: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    module.ckks_mul_pt_vec_assign(dst, scale, scratch)?;
    module.ckks_add_pt_vec_assign(dst, offset, scratch)
}

pub(crate) fn ckks_mul_sub_ct_tmp_bytes_derived<BE, R, A, B, T>(module: &Module<BE>, res: &R, a: &A, b: &B, tsk: &T) -> usize
where
    BE: CKKSMulSubImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    T: GGLWEInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_tmp_bytes(res, a, b, tsk)
            .max(module.ckks_sub_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_sub_pt_vec_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulSubImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_vec_tmp_bytes(res, a, b)
            .max(module.ckks_sub_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_sub_pt_const_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulSubImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_const_tmp_bytes(res, a, b)
            .max(module.ckks_sub_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_sub_ct_into_derived<BE, Dst, A, B, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    b: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulSubImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetTensorKey<BE>,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_into(tmp, a, b, tsk, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_sub_pt_vec_into_derived<BE, Dst, A, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulSubImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_vec_into(tmp, a, pt, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_sub_pt_const_into_derived<BE, Dst, A, P>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    pt_coeff: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulSubImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    mul_then_combine(
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}
