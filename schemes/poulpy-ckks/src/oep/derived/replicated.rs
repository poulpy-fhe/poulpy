use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSModuleInfos, ReplicatedLayout, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout,
    api::{CKKSAddOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSPow2Ops, CKKSRotateOps, CKKSRoundOps},
    layouts::{
        CKKSCiphertextViewMut, ReplicatedConversion, ReplicatedTransform, ReplicatedTransformWorkspace,
        ScratchArenaTakeCKKSCiphertext,
    },
};

pub(crate) fn invalid(op: &'static str, reason: &str) -> crate::CKKSError {
    CKKSCompositionError::InvalidOperand {
        op,
        reason: reason.into(),
    }
    .into()
}

pub(crate) fn check_destination<Dst: GLWEInfos, Src: GLWEInfos>(op: &'static str, dst: &Dst, src: &Src) -> Result<()> {
    if dst.rank() != src.rank() {
        return Err(invalid(op, "source and destination ranks differ"));
    }
    if dst.n() != src.n() {
        return Err(invalid(op, "source and destination degrees differ"));
    }
    Ok(())
}

/// The working layout of a stage on `ct`: `extra_bits` under its scale and
/// room for a trace's division by `2^trace_steps`.
pub(crate) fn working_layout<C: GLWEInfos>(ct: &C, extra_bits: usize, trace_steps: usize) -> GLWELayout {
    GLWELayout {
        k: (ct.k().as_usize() + extra_bits + trace_steps).into(),
        ..ct.glwe_layout()
    }
}

/// Wide stage onto a replicated layout: `dst = M(src)` computed `extra_bits`
/// under `src`'s scale, then rounded with `layout`'s shaping to `log_delta`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wide_onto_replicated<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    transform: &ReplicatedTransform<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    layout: ReplicatedLayout,
    log_delta: usize,
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + CKKSCopyOps<BE> + CKKSRoundOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    let working = working_layout(src, extra_bits, 0);
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut acc, mut scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());

    module.ckks_copy(&mut x, src, &mut scratch)?;
    x.set_log_delta(src.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_transform(module, &mut acc, &x, transform, workspace, keys, &mut scratch)?;
    acc.set_slot_layout(SlotLayout::Replicated(layout));
    dst.set_k((acc.log_budget() + log_delta).into());
    module.ckks_round(dst, &acc, &mut scratch)
}

/// Wide stage gathering onto period `k`: `dst = T(M(src))` computed `extra_bits` under `src`'s scale, where the normalized trace `T`
/// sums the rotations by `trace` after dividing by `2^trace.len()`, which lifts
/// the modulus so that its multiples vanish. Rounds plainly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wide_gather_trace<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    transform: &ReplicatedTransform<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    trace: &[i64],
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    let working = working_layout(src, extra_bits, trace.len());
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut rotated, mut scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());

    module.ckks_copy(&mut x, src, &mut scratch)?;
    x.set_log_delta(src.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_transform(module, &mut acc, &x, transform, workspace, keys, &mut scratch)?;
    normalized_trace(module, &mut acc, trace, keys, &mut rotated, &mut scratch)?;
    module.ckks_copy(dst, &acc, &mut scratch)
}

/// `ct = (1/2^L) Σ_(i < 2^L) rot_(i·step)(ct)` by `L` rotations by `step, 2·step, ...`.
/// The division comes first and lifts the modulus: the trace of the modulus'
/// multiples lies in `2^L·R`, so they vanish on the torus. A trace closing the
/// slot cycle, `step·2^L = N/2`, leaves a polynomial in `X^(2^L)`.
pub(crate) fn normalized_trace<BE, H>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextViewMut<'_, BE>,
    shifts: &[i64],
    keys: &H,
    rotated: &mut CKKSCiphertextViewMut<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSPow2Ops<BE>,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_div_pow2_assign(shifts.len(), ct)?;
    for &shift in shifts {
        module.ckks_rotate_into(shift, rotated, &*ct, keys, scratch)?;
        module.ckks_add_assign(ct, &*rotated, scratch)?;
    }
    if shifts
        .first()
        .is_some_and(|&step| (step as usize) << shifts.len() == module.ckks_max_slots())
    {
        ct.set_log_sparsity(shifts.len());
    }
    Ok(())
}

/// `acc = M(x)` by the prepared BSGS transform `M`.
pub(crate) fn apply_transform<BE, H>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertextViewMut<'_, BE>,
    x: &CKKSCiphertextViewMut<'_, BE>,
    transform: &ReplicatedTransform<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    check_workspace("replicated_transform", workspace, &x.glwe_layout(), transform)?;
    module.ckks_linear_transformation_baby_steps_prepare(&mut workspace.babies, x, keys, scratch)?;
    module.ckks_linear_transformation_eval_into(acc, x, &workspace.babies, &transform.lt, keys, scratch)
}

pub(crate) fn check_workspace<BE: Backend>(
    op: &'static str,
    workspace: &ReplicatedTransformWorkspace<BE>,
    input: &GLWELayout,
    transform: &ReplicatedTransform<BE>,
) -> Result<()> {
    if workspace.input != *input || !workspace.babies.baby_steps().eq(transform.lt.baby_steps().iter().copied()) {
        return Err(invalid(
            op,
            "workspace does not match the transform input layout or rotations",
        ));
    }
    Ok(())
}

/// Scratch of a wide stage of `transform` on `wide`, beside its ciphertexts.
pub(crate) fn transform_tmp_bytes<BE, K>(module: &Module<BE>, wide: &CKKSLayout, key_infos: &K) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    K: GGLWEInfos,
{
    module
        .ckks_linear_transformation_baby_steps_prepare_tmp_bytes(
            wide,
            &poulpy_core::layouts::LinearTransformationPlan::default(),
            key_infos,
        )
        .max(module.ckks_linear_transformation_eval_tmp_bytes(
            wide,
            &crate::api::ckks_plaintext_diagonal_layout(wide),
            &poulpy_core::layouts::LinearTransformationPlan::default(),
            key_infos,
            key_infos,
        ))
}

pub(crate) fn ct_bytes<BE: Backend>(layout: &GLWELayout) -> usize {
    BE::scratch_aligned(BE::vec_znx_bytes_of(
        layout.n.as_usize(),
        layout.rank.as_usize() + 1,
        layout.size(),
    ))
}

/// The normalized trace of `ct` onto the subring of `skip`, computed
/// `extra_bits` under its scale and rounded back to its width, with shaping
/// when `ct` is in a replicated layout. The average of the copies keeps the
/// message's scale and budget.
pub(crate) fn ckks_replicated_conversion_tmp_bytes_derived<BE, C, K>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
    conv: &ReplicatedConversion<BE>,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>,
    C: CKKSCtBounds,
    K: GGLWEInfos,
{
    let layout = working_layout(ct_infos, conv.extra_bits, conv.trace.len());
    let wide = CKKSLayout {
        glwe_layout: layout,
        meta: ct_infos.meta(),
    };
    let ct_bytes = BE::scratch_aligned(BE::vec_znx_bytes_of(
        layout.n.as_usize(),
        layout.rank.as_usize() + 1,
        layout.size(),
    ));
    let ops = transform_tmp_bytes(module, &wide, key_infos)
        .max(module.ckks_rotate_tmp_bytes(&wide, key_infos))
        .max(module.ckks_add_tmp_bytes(layout.size()))
        .max(module.ckks_copy_tmp_bytes(&wide, ct_infos))
        .max(module.ckks_round_tmp_bytes(ct_infos, &wide))
        .max(module.ckks_div_pow2_tmp_bytes(layout.size()));
    3 * ct_bytes + ops
}

pub(crate) fn ckks_to_replicated_derived<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    conv: &ReplicatedConversion<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    const OP: &str = "ckks_to_replicated";
    check_destination(OP, dst, src)?;
    if src.slot_layout() != SlotLayout::Full {
        return Err(invalid(OP, "the source must have the full slot layout"));
    }
    if conv.log_delta > src.log_delta() + conv.extra_bits {
        return Err(invalid(OP, "the replicated scale exceeds the source's working scale"));
    }
    check_workspace(OP, workspace, &working_layout(src, conv.extra_bits, 0), &conv.to)?;
    wide_onto_replicated(
        module,
        dst,
        src,
        &conv.to,
        workspace,
        conv.layout,
        conv.log_delta,
        conv.extra_bits,
        keys,
        scratch,
    )
}

pub(crate) fn ckks_from_replicated_derived<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    conv: &ReplicatedConversion<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    const OP: &str = "ckks_from_replicated";
    check_destination(OP, dst, src)?;
    if src.slot_layout() != SlotLayout::Replicated(conv.layout) {
        return Err(invalid(OP, "the source must have the conversion's replicated layout"));
    }
    let Some(from) = &conv.from else {
        return Err(invalid(OP, "the gather needs the period to divide the batch size"));
    };
    check_workspace(OP, workspace, &working_layout(src, conv.extra_bits, 0), from)?;
    wide_gather_trace(module, dst, src, from, workspace, &conv.trace, conv.extra_bits, keys, scratch)
}
