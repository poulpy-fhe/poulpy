//! Multiply-accumulate and affine forms compose the selected products and additions.
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos},
};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps, CKKSSubOps},
    ckks_ensure, ckks_mul_pt_checks, ensure_plaintext_operand, get_mul_ct_params,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext},
    mul_ct_log_delta, mul_pt_log_delta,
    oep::{CKKSAffineImpl, CKKSMulAddImpl, CKKSMulSubImpl},
    reference::mul::{product_key_k, tensor_key_at},
};

/// Checks the arena against the caller's query `bytes`, then carves a `dst`-shaped temporary in a scratch scope, runs
/// the product into it and folds it into `dst`. Callers run the product's operand checks first.
fn mul_then_combine<BE, Dst, MulF, CombineF, Rg: Ring>(
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
    MulF: for<'t> FnOnce(&mut CKKSCiphertextViewMut<'t, BE, Rg>, &mut ScratchArena<'t, BE>) -> Result<()>,
    CombineF: for<'t> FnOnce(&mut Dst, &CKKSCiphertextViewMut<'t, BE, Rg>, &mut ScratchArena<'t, BE>) -> Result<()>,
{
    ckks_ensure!(scratch.available() >= bytes, "{op}: scratch below its query");
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        mul(&mut tmp, &mut scratch_local)?;
        combine(dst, &tmp, &mut scratch_local)
    })
}

/// The checks the addition of `offset` runs on `dst` once the product left it at `log_budget`.
fn check_affine_offset<R: CKKSCtBounds, P: CKKSInfos>(op: &'static str, dst: &R, log_budget: usize, offset: &P) -> Result<()> {
    ensure_plaintext_operand(op, dst.base2k().as_usize(), dst.n().as_usize(), log_budget, offset)
}

/// The refusal the addition of `offset` to the product of `a`, at scale `2^log_delta`, would raise: on a replicated
/// layout, an offset of a finer scale. Refused under `op`, before the product writes.
fn check_affine_offset_scale<A: CKKSCtBounds, P: CKKSInfos>(op: &'static str, a: &A, log_delta: usize, offset: &P) -> Result<()> {
    if offset.log_delta() > log_delta {
        crate::layouts::validation::reject_replicated_scales(
            op,
            crate::GLWEPlaintextInfos::slot_layout(a),
            log_delta,
            offset.log_delta(),
        )?;
    }
    Ok(())
}

/// The checks the accumulation of a fused multiply-accumulate runs before the product: one slot
/// layout for the operands and the accumulator, and on a replicated layout the accumulator at the
/// product's scale, which the sum would otherwise round plainly. Refused under `op`, before the product's own checks.
/// Returns the product's slot layout.
fn check_accumulation<const K: usize>(
    op: &'static str,
    layouts: [crate::SlotLayout; K],
    dst_log_delta: usize,
    product_log_delta: usize,
) -> Result<crate::SlotLayout> {
    let slot_layout = crate::layouts::validation::product_layout(op, &layouts)?;
    crate::layouts::validation::reject_replicated_scales(op, slot_layout, dst_log_delta, product_log_delta)?;
    Ok(slot_layout)
}

pub(crate) fn ckks_mul_add_tmp_bytes_derived<BE, R, A, B, T, Rg: Ring>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    b: &B,
    tsk: &T,
) -> usize
where
    BE: CKKSMulAddImpl<Rg>,
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

pub(crate) fn ckks_mul_add_pt_tmp_bytes_derived<BE, R, A, P, Rg: Ring>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulAddImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_tmp_bytes(res, a, b)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_add_into_derived<BE, Dst, A, B, H, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    b: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulAddImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    const OP: &str = "ckks_mul_add_into";
    crate::ensure_operand_degrees(OP, module, [dst.n(), a.n(), b.n()])?;
    let slot_layout = check_accumulation(
        OP,
        [
            crate::GLWEPlaintextInfos::slot_layout(&*dst),
            crate::GLWEPlaintextInfos::slot_layout(a),
            crate::GLWEPlaintextInfos::slot_layout(b),
        ],
        dst.log_delta(),
        mul_ct_log_delta(a.log_delta(), b.log_delta()),
    )?;
    let (log_budget, log_delta, cnv_offset) = get_mul_ct_params(OP, &*dst, a, b)?;
    let key_k = product_key_k(
        &*dst,
        &[a.base2k(), b.base2k()],
        a.k().max(b.k()),
        cnv_offset,
        log_budget + log_delta,
        crate::shaping::product_depth(slot_layout, log_delta),
        tsk,
    );
    let key = tensor_key_at(OP, tsk, key_k)?;
    let bytes = ckks_mul_add_tmp_bytes_derived(module, &*dst, a, b, &key);
    mul_then_combine(
        OP,
        bytes,
        dst,
        scratch,
        |tmp, s| module.ckks_mul_into(tmp, a, b, tsk, s),
        |dst, tmp, s| module.ckks_add_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_add_pt_into_derived<BE, Dst, A, P, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulAddImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
{
    const OP: &str = "ckks_mul_add_pt_into";
    crate::ensure_operand_degrees(OP, module, [dst.n(), a.n()])?;
    check_accumulation(
        OP,
        [
            crate::GLWEPlaintextInfos::slot_layout(&*dst),
            crate::GLWEPlaintextInfos::slot_layout(a),
            crate::layouts::validation::factor_layout(a, pt),
        ],
        dst.log_delta(),
        mul_pt_log_delta(a.log_delta()),
    )?;
    ckks_mul_pt_checks(OP, &*dst, a, pt)?;
    let bytes = ckks_mul_add_pt_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_into(tmp, a, pt, s),
        |dst, tmp, s| module.ckks_add_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_affine_pt_tmp_bytes_derived<BE, R, A, S, Rg: Ring>(module: &Module<BE>, res: &R, a: &A, scale: &S) -> usize
where
    BE: CKKSAffineImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    S: CKKSInfos,
{
    module
        .ckks_mul_pt_tmp_bytes(res, a, scale)
        .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_affine_pt_into_derived<BE, Dst, A, S, P, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    scale: &S,
    offset: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
    P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
{
    const OP: &str = "ckks_affine_pt_into";
    crate::ensure_operand_degrees(OP, module, [dst.n(), a.n()])?;
    crate::layouts::validation::product_layout(
        OP,
        &[
            crate::GLWEPlaintextInfos::slot_layout(a),
            crate::layouts::validation::factor_layout(a, scale),
            crate::GLWEPlaintextInfos::slot_layout(offset),
        ],
    )?;
    let (log_budget, log_delta, _) = ckks_mul_pt_checks(OP, &*dst, a, scale)?;
    check_affine_offset(OP, &*dst, log_budget, offset)?;
    check_affine_offset_scale(OP, a, log_delta, offset)?;
    module.ckks_mul_pt_into(dst, a, scale, scratch)?;
    module.ckks_add_assign(dst, offset, scratch)
}

pub(crate) fn ckks_affine_pt_assign_derived<BE, Dst, S, P, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    scale: &S,
    offset: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSAffineImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
    P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
{
    const OP: &str = "ckks_affine_pt_assign";
    crate::ensure_operand_degrees(OP, module, [dst.n()])?;
    crate::layouts::validation::product_layout(
        OP,
        &[
            crate::GLWEPlaintextInfos::slot_layout(dst),
            crate::layouts::validation::factor_layout(&*dst, scale),
            crate::GLWEPlaintextInfos::slot_layout(offset),
        ],
    )?;
    let (log_budget, log_delta, _) = ckks_mul_pt_checks(OP, &*dst, &*dst, scale)?;
    check_affine_offset(OP, &*dst, log_budget, offset)?;
    check_affine_offset_scale(OP, &*dst, log_delta, offset)?;
    module.ckks_mul_pt_assign(dst, scale, scratch)?;
    module.ckks_add_assign(dst, offset, scratch)
}

pub(crate) fn ckks_mul_sub_tmp_bytes_derived<BE, R, A, B, T, Rg: Ring>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    b: &B,
    tsk: &T,
) -> usize
where
    BE: CKKSMulSubImpl<Rg>,
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

pub(crate) fn ckks_mul_sub_pt_tmp_bytes_derived<BE, R, A, P, Rg: Ring>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSMulSubImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_tmp_bytes(res, a, b)
            .max(module.ckks_sub_tmp_bytes(res.size()))
}

pub(crate) fn ckks_mul_sub_into_derived<BE, Dst, A, B, H, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    b: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulSubImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    const OP: &str = "ckks_mul_sub_into";
    crate::ensure_operand_degrees(OP, module, [dst.n(), a.n(), b.n()])?;
    let slot_layout = check_accumulation(
        OP,
        [
            crate::GLWEPlaintextInfos::slot_layout(&*dst),
            crate::GLWEPlaintextInfos::slot_layout(a),
            crate::GLWEPlaintextInfos::slot_layout(b),
        ],
        dst.log_delta(),
        mul_ct_log_delta(a.log_delta(), b.log_delta()),
    )?;
    let (log_budget, log_delta, cnv_offset) = get_mul_ct_params(OP, &*dst, a, b)?;
    let key_k = product_key_k(
        &*dst,
        &[a.base2k(), b.base2k()],
        a.k().max(b.k()),
        cnv_offset,
        log_budget + log_delta,
        crate::shaping::product_depth(slot_layout, log_delta),
        tsk,
    );
    let key = tensor_key_at(OP, tsk, key_k)?;
    let bytes = ckks_mul_sub_tmp_bytes_derived(module, &*dst, a, b, &key);
    mul_then_combine(
        OP,
        bytes,
        dst,
        scratch,
        |tmp, s| module.ckks_mul_into(tmp, a, b, tsk, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}

pub(crate) fn ckks_mul_sub_pt_into_derived<BE, Dst, A, P, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &A,
    pt: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSMulSubImpl<Rg>,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
{
    const OP: &str = "ckks_mul_sub_pt_into";
    crate::ensure_operand_degrees(OP, module, [dst.n(), a.n()])?;
    check_accumulation(
        OP,
        [
            crate::GLWEPlaintextInfos::slot_layout(&*dst),
            crate::GLWEPlaintextInfos::slot_layout(a),
            crate::layouts::validation::factor_layout(a, pt),
        ],
        dst.log_delta(),
        mul_pt_log_delta(a.log_delta()),
    )?;
    ckks_mul_pt_checks(OP, &*dst, a, pt)?;
    let bytes = ckks_mul_sub_pt_tmp_bytes_derived(module, &*dst, a, pt);
    mul_then_combine(
        OP,
        bytes,
        dst,
        scratch,
        |tmp, s| module.ckks_mul_pt_into(tmp, a, pt, s),
        |dst, tmp, s| module.ckks_sub_assign(dst, tmp, s),
    )
}
