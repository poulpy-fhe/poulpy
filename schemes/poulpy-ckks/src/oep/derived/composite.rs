//! Multiply-accumulate and affine forms compose the selected products and additions.
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps, CKKSSubOps},
    ckks_ensure, ckks_mul_pt_const_checks, ckks_mul_pt_vec_checks, ensure_plaintext_const_operand, ensure_plaintext_vec_operand,
    get_mul_ct_params,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext},
    oep::{CKKSAffineImpl, CKKSMulAddImpl, CKKSMulSubImpl},
    reference::mul::tensor_key_at,
};

/// Checks the arena against the caller's query `bytes`, then carves a `dst`-shaped temporary in a scratch scope, runs
/// the product into it and folds it into `dst`. Callers run the product's operand checks first.
fn mul_then_combine<BE, Dst, MulF, CombineF>(
    op: &'static str,
    bytes: usize,
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
    ckks_ensure!(scratch.available() >= bytes, "{op}: scratch below its query");
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        mul(&mut tmp, &mut scratch_local)?;
        combine(dst, &tmp, &mut scratch_local)
    })
}

/// The checks the addition of coefficient `offset_coeff` of `cst` at coefficient 0 runs on `dst` once the product left
/// it at `log_budget`.
fn check_affine_offset_const<R: CKKSCtBounds, P: CKKSInfos>(
    op: &'static str,
    dst: &R,
    log_budget: usize,
    cst: &P,
    offset_coeff: usize,
) -> Result<()> {
    ensure_plaintext_const_operand(
        op,
        dst.base2k().as_usize(),
        dst.n().as_usize(),
        0,
        log_budget,
        cst,
        offset_coeff,
    )
}

/// The checks the addition of `offset` runs on `dst` once the product left it at `log_budget`.
fn check_affine_offset_vec<BE: Backend, R: CKKSCtBounds, P: CKKSInfos>(
    op: &'static str,
    dst: &R,
    log_budget: usize,
    offset: &P,
) -> Result<()> {
    ensure_plaintext_vec_operand(
        op,
        dst.base2k().as_usize(),
        dst.n().as_usize(),
        log_budget,
        offset,
        BE::MIN_DEGREE,
    )
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
    const OP: &str = "ckks_mul_add_ct_into";
    let key = tensor_key_at(OP, tsk, a.k().max(b.k()))?;
    get_mul_ct_params(OP, &*dst, a, b)?;
    let bytes = ckks_mul_add_ct_tmp_bytes_derived(module, &*dst, a, b, &key);
    mul_then_combine(
        OP,
        bytes,
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
    const OP: &str = "ckks_mul_add_pt_vec_into";
    ckks_mul_pt_vec_checks::<BE, _, _, _>(OP, &*dst, a, pt)?;
    let bytes = ckks_mul_add_pt_vec_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
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
    const OP: &str = "ckks_mul_add_pt_const_into";
    ckks_mul_pt_const_checks(OP, &*dst, a, pt, pt_coeff)?;
    let bytes = ckks_mul_add_pt_const_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
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
    const OP: &str = "ckks_affine_pt_const_into";
    let (log_budget, _, _) = ckks_mul_pt_const_checks(OP, &*dst, a, affine_const, scale_coeff)?;
    check_affine_offset_const(OP, &*dst, log_budget, affine_const, offset_coeff)?;
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
    const OP: &str = "ckks_affine_pt_const_assign";
    let (log_budget, _, _) = ckks_mul_pt_const_checks(OP, &*dst, &*dst, affine_const, scale_coeff)?;
    check_affine_offset_const(OP, &*dst, log_budget, affine_const, offset_coeff)?;
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
    const OP: &str = "ckks_affine_pt_vec_into";
    let (log_budget, _, _) = ckks_mul_pt_vec_checks::<BE, _, _, _>(OP, &*dst, a, scale)?;
    check_affine_offset_vec::<BE, _, _>(OP, &*dst, log_budget, offset)?;
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
    const OP: &str = "ckks_affine_pt_vec_assign";
    let (log_budget, _, _) = ckks_mul_pt_vec_checks::<BE, _, _, _>(OP, &*dst, &*dst, scale)?;
    check_affine_offset_vec::<BE, _, _>(OP, &*dst, log_budget, offset)?;
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
    const OP: &str = "ckks_mul_sub_ct_into";
    let key = tensor_key_at(OP, tsk, a.k().max(b.k()))?;
    get_mul_ct_params(OP, &*dst, a, b)?;
    let bytes = ckks_mul_sub_ct_tmp_bytes_derived(module, &*dst, a, b, &key);
    mul_then_combine(
        OP,
        bytes,
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
    const OP: &str = "ckks_mul_sub_pt_vec_into";
    ckks_mul_pt_vec_checks::<BE, _, _, _>(OP, &*dst, a, pt)?;
    let bytes = ckks_mul_sub_pt_vec_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
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
    const OP: &str = "ckks_mul_sub_pt_const_into";
    ckks_mul_pt_const_checks(OP, &*dst, a, pt, pt_coeff)?;
    let bytes = ckks_mul_sub_pt_const_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}
