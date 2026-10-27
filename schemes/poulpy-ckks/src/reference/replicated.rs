use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECIEmbed, GLWECITrace, GLWECopy, GLWEKeyswitch, GLWEShift, GLWETrace,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWELayout,
        GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, Scale, SetK,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSModuleInfos, GLWEPlaintextInfos, ReplicatedLayout,
    SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSCIRingMapOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSPow2Ops, CKKSReplicatedBootstrapOps, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        BootstrappingKeys, CKKSCiphertextViewMut, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedTransform, ReplicatedTransformWorkspace, ScratchArenaTakeCKKS,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
    reference::bootstrapping::ckks_bootstrap_mod_up_from_mut,
};

fn invalid(op: &'static str, reason: &str) -> crate::CKKSError {
    CKKSCompositionError::InvalidOperand {
        op,
        reason: reason.into(),
    }
    .into()
}

fn check_destination<Dst: GLWEInfos, Src: GLWEInfos>(op: &'static str, dst: &Dst, src: &Src) -> Result<()> {
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
fn working_layout<C: GLWEInfos>(ct: &C, extra_bits: usize, trace_steps: usize) -> GLWELayout {
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
    H: GetAutomorphismKey<BE>,
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
    module.ckks_round_into(dst, &acc, &mut scratch)
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
    H: GetAutomorphismKey<BE>,
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
fn apply_transform<BE, H>(
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
    H: GetAutomorphismKey<BE>,
{
    check_workspace("replicated_transform", workspace, &x.glwe_layout(), transform)?;
    module.ckks_prepare_linear_transformation_baby_steps(&mut workspace.babies, x, keys, scratch)?;
    module.ckks_eval_linear_transformation_into(acc, x, &workspace.babies, &transform.lt, keys, scratch)
}

fn check_workspace<BE: Backend>(
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
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(
            wide,
            &poulpy_core::layouts::LinearTransformationPlan::default(),
            key_infos,
        )
        .max(module.ckks_eval_linear_transformation_tmp_bytes(
            wide,
            &poulpy_core::layouts::LinearTransformationPlan::default(),
            key_infos,
        ))
}

fn ct_bytes<BE: Backend>(layout: &GLWELayout) -> usize {
    BE::scratch_aligned(BE::bytes_of_vec_znx(
        layout.n.as_usize(),
        layout.rank.as_usize() + 1,
        layout.size(),
    ))
}

/// The normalized trace of `ct` onto the subring of `skip`, computed
/// `extra_bits` under its scale and rounded back to its width, with shaping
/// when `ct` is in a replicated layout. The average of the copies keeps the
/// message's scale and budget.
fn trace_wide<BE, H>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextViewMut<'_, BE>,
    skip: usize,
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWETrace<BE> + CKKSCopyOps<BE> + CKKSRoundOps<BE>,
    H: GetAutomorphismKey<BE>,
{
    let working = working_layout(&*ct, extra_bits, 0);
    let (mut wide, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, ct.meta());
    module.ckks_copy(&mut wide, &*ct, &mut scratch)?;
    wide.set_log_delta(ct.log_delta() + extra_bits);
    let log_n = wide.n().log2();
    module.glwe_trace_assign(&mut wide, skip..log_n, false, true, keys, &mut scratch);
    let log_delta = ct.log_delta();
    ct.set_k((wide.log_budget() + log_delta).into());
    module.ckks_round_into(ct, &wide, &mut scratch)
}

/// `CoeffsToSlots` as a wide stage: `w = M(src)` computed `extra_bits` under
/// `src`'s scale in `layout`, then `re = w + conj(w)` and, unless the parts are
/// packed in `layout`, `im = i·(conj(w) − w)`, each rounded with shaping to
/// `src`'s scale.
#[allow(clippy::too_many_arguments)]
fn coeffs_to_slots<BE, H>(
    module: &Module<BE>,
    re: &mut CKKSCiphertextViewMut<'_, BE>,
    im: Option<&mut CKKSCiphertextViewMut<'_, BE>>,
    src: &CKKSCiphertextViewMut<'_, BE>,
    transform: &ReplicatedTransform<BE>,
    workspace: &mut ReplicatedTransformWorkspace<BE>,
    layout: ReplicatedLayout,
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRoundOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>,
    H: GetAutomorphismKey<BE>,
{
    let working = working_layout(src, extra_bits, 0);
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut w, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut conj, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut part, mut scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());

    module.ckks_copy(&mut x, src, &mut scratch)?;
    x.set_log_delta(src.log_delta() + extra_bits);
    apply_transform(module, &mut w, &x, transform, workspace, keys, &mut scratch)?;
    w.set_slot_layout(SlotLayout::Replicated(layout));
    module.ckks_conjugate_into(&mut conj, &w, keys, &mut scratch)?;
    module.ckks_add_into(&mut part, &w, &conj, &mut scratch)?;
    part.set_slots(SlotsKind::Real);
    re.set_k((part.log_budget() + src.log_delta()).into());
    module.ckks_round_into(re, &part, &mut scratch)?;
    if let Some(im) = im {
        module.ckks_sub_into(&mut part, &conj, &w, &mut scratch)?;
        module.ckks_mul_i_assign(&mut part, &mut scratch)?;
        part.set_slots(SlotsKind::Real);
        im.set_k((part.log_budget() + src.log_delta()).into());
        module.ckks_round_into(im, &part, &mut scratch)?;
    }
    Ok(())
}

pub(crate) fn ckks_replicated_conversion_tmp_bytes<BE, C, K>(
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
    let ct_bytes = BE::scratch_aligned(BE::bytes_of_vec_znx(
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

pub(crate) fn ckks_to_replicated_into<BE, Dst, Src, H>(
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
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_to_replicated_into";
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

pub(crate) fn ckks_from_replicated_into<BE, Dst, Src, H>(
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
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_from_replicated_into";
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

pub(crate) fn ckks_replicated_bootstrap_tmp_bytes<BE, C, I, F, A, T, S>(
    module: &Module<BE>,
    ct_out: &C,
    ct_in: &I,
    ctx: &ReplicatedBootstrapping<BE, F>,
    atk_infos: &A,
    tsk_infos: &T,
    encapsulation_infos: Option<(&S, &S)>,
) -> usize
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSEvalModOps<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWETrace<BE>,
    C: CKKSCtBounds,
    I: CKKSCtBounds,
    A: GGLWEInfos,
    T: GGLWEInfos,
    S: GGLWEInfos,
{
    let plan = ctx.plan;
    let at = |k: usize| GLWELayout {
        k: k.into(),
        ..ct_out.glwe_layout()
    };
    let ckks = |glwe_layout: GLWELayout, slot_layout| CKKSLayout {
        glwe_layout,
        meta: GLWEPlaintextMeta {
            slot_layout,
            ..ct_out.meta()
        },
    };
    let replicated = SlotLayout::Replicated(plan.layout);
    let k_boot = plan.k_boot(ct_out.k().as_usize());
    let input = at(plan.k_in());
    let low = working_layout(&input, plan.extra_bits, 0);
    let raised = at(k_boot);
    let high = working_layout(&raised, plan.extra_bits, 0);
    let boot = ckks(at(k_boot), replicated);
    let wide = |layout: GLWELayout| {
        [replicated, SlotLayout::Full]
            .into_iter()
            .map(|slot_layout| {
                let wide = ckks(layout, slot_layout);
                transform_tmp_bytes(module, &wide, atk_infos)
                    .max(module.ckks_rotate_tmp_bytes(&wide, atk_infos))
                    .max(module.ckks_conjugate_tmp_bytes(&wide, atk_infos))
                    .max(module.ckks_copy_tmp_bytes(&wide, &wide))
                    .max(module.ckks_round_tmp_bytes(&boot, &wide))
                    .max(module.ckks_add_tmp_bytes(layout.size()))
                    .max(module.ckks_sub_tmp_bytes(layout.size()))
                    .max(module.ckks_mul_i_tmp_bytes(layout.size()))
                    .max(module.glwe_trace_assign_tmp_bytes(&layout, atk_infos))
            })
            .max()
            .unwrap_or(0)
    };
    let input_ckks = ckks(input, replicated);
    let mod_up = match encapsulation_infos {
        Some((dense_to_sparse, sparse_to_dense)) => {
            BE::ckks_encapsulated_mod_up_tmp_bytes(module, &boot, &input_ckks, dense_to_sparse, sparse_to_dense)
        }
        None => module.glwe_shift_tmp_bytes(raised.size()),
    }
    .max(module.ckks_mul_pow2_tmp_bytes(input.size()));
    let eval_mod = module
        .ckks_eval_mod_tmp_bytes(&boot, &boot, &ctx.eval_mod, tsk_infos)
        .max(module.ckks_rotate_tmp_bytes(&boot, atk_infos))
        .max(module.ckks_mul_i_tmp_bytes(raised.size()))
        .max(module.ckks_add_tmp_bytes(raised.size()))
        .max(module.ckks_round_tmp_bytes(&ckks(ct_out.glwe_layout(), replicated), &boot));
    let lift = plan.log_delta.saturating_sub(ct_in.log_delta());
    let narrow = at(ct_in.k().as_usize() + lift);
    let front = ct_bytes::<BE>(&input) + ct_bytes::<BE>(&narrow) + 3 * ct_bytes::<BE>(&low) + wide(low).max(mod_up);
    let front = front.max(ct_bytes::<BE>(&input) + ct_bytes::<BE>(&low) + wide(low));
    let back = 3 * ct_bytes::<BE>(&raised) + (4 * ct_bytes::<BE>(&high) + wide(high)).max(eval_mod);
    ct_bytes::<BE>(&raised) + front.max(back)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_replicated_bootstrap_into<BE, Dst, Src, F, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    ctx: &ReplicatedBootstrapping<BE, F>,
    workspace: &mut ReplicatedBootstrapWorkspace<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSEvalModOps<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWETrace<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: BootstrappingKeys<BE>,
{
    const OP: &str = "ckks_replicated_bootstrap_into";
    check_destination(OP, dst, src)?;
    let plan = ctx.plan;
    if src.slot_layout() != SlotLayout::Replicated(plan.layout) {
        return Err(invalid(OP, "the source must have the bootstrapping's replicated layout"));
    }
    if src.rank().as_usize() != 1 || dst.rank().as_usize() != 1 {
        return Err(invalid(OP, "bootstrapping supports rank-1 ciphertexts only"));
    }
    let lift = match src.log_delta() {
        d if d == plan.log_delta => 0,
        d if d == plan.level_log_delta && d < plan.log_delta => plan.log_delta - d,
        _ => return Err(invalid(OP, "the source must be at the bootstrapping or the leveled scale")),
    };
    let k_in = plan.k_in();
    if src.log_budget() + plan.log_delta < k_in {
        return Err(invalid(OP, "the source has less budget than the bootstrapping input"));
    }
    let rotation_keys = keys.rotation_keys();
    let at = |k: usize| GLWELayout {
        k: k.into(),
        ..src.glwe_layout()
    };
    let k_boot = plan.k_boot(dst.k().as_usize());
    check_workspace(OP, &workspace.s2c, &at(k_in + plan.extra_bits), &ctx.s2c)?;
    check_workspace(OP, &workspace.c2s, &at(k_boot + plan.extra_bits), &ctx.c2s)?;
    let mut raised_meta = src.meta();
    raised_meta.slot_layout = SlotLayout::Full;
    let (mut raised, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&at(k_boot), raised_meta);

    scratch.scope(|scratch| -> Result<()> {
        let input = at(k_in);
        let (mut coeffs, scratch) = scratch.take_ckks_ciphertext_scratch(&input, src.meta());
        let (mut narrow, scratch) = scratch.take_ckks_ciphertext_scratch(&at(src.k().as_usize() + lift), src.meta());
        let low = working_layout(&input, plan.extra_bits, 0);
        let (mut x, scratch) = scratch.take_ckks_ciphertext_scratch(&low, src.meta());
        let (mut acc, mut scratch) = scratch.take_ckks_ciphertext_scratch(&low, src.meta());

        // 1. At `Δ_b`, a leveled input relabelled at the same torus value, then
        //    reduced to the input width.
        module.ckks_copy(&mut narrow, src, &mut scratch)?;
        if lift > 0 {
            narrow.set_log_delta(plan.log_delta);
        }
        module.ckks_copy(&mut coeffs, &narrow, &mut scratch)?;

        // 2. The copies of the period.
        trace_wide(
            module,
            &mut coeffs,
            ctx.bottom_skip,
            plan.extra_bits,
            rotation_keys,
            &mut scratch,
        )?;
        module.ckks_copy(&mut x, &coeffs, &mut scratch)?;

        // 3. SlotsToCoeffs, rounded plainly to the message ratio.
        x.set_log_delta(x.log_delta() + plan.extra_bits);
        x.set_slot_layout(SlotLayout::Full);
        apply_transform(
            module,
            &mut acc,
            &x,
            &ctx.s2c,
            &mut workspace.s2c,
            rotation_keys,
            &mut scratch,
        )?;
        coeffs.set_k((acc.log_budget() + plan.log_delta).into());
        module.ckks_round_into(&mut coeffs, &acc, &mut scratch)?;
        if coeffs.log_budget() != plan.eval_mod.log_msg_ratio {
            return Err(invalid(OP, "the coefficient message does not have the EvalMod message ratio"));
        }

        // 4. ModUp: `I·q_b + Δ_b·m` at the scale `q_b`, lifted.
        let log_q_b = coeffs.k().as_usize();
        raised.set_k(k_boot.into());
        ckks_bootstrap_mod_up_from_mut(
            module,
            &mut raised,
            &mut coeffs,
            None,
            plan.mod_up_extra_bits,
            None,
            keys,
            &mut scratch,
        )?;
        raised.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_q_b + plan.mod_up_extra_bits),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        });
        Ok(())
    })?;

    // 5. The coefficients of the subring, onto the data slots.
    let s_raised = raised.log_delta();
    let high = at(raised.k().as_usize());
    let (mut re, scratch) = scratch.take_ckks_ciphertext_scratch(&high, raised.meta());
    let (mut im, scratch) = scratch.take_ckks_ciphertext_scratch(&high, raised.meta());
    let (mut out, mut scratch) = scratch.take_ckks_ciphertext_scratch(&high, raised.meta());
    trace_wide(
        module,
        &mut raised,
        ctx.top_skip,
        plan.extra_bits,
        rotation_keys,
        &mut scratch,
    )?;
    raised.set_log_sparsity(ctx.top_sparsity);
    let packed = ctx.packed.unwrap_or(plan.layout);
    coeffs_to_slots(
        module,
        &mut re,
        ctx.packed.is_none().then_some(&mut im),
        &raised,
        &ctx.c2s,
        &mut workspace.c2s,
        packed,
        plan.extra_bits,
        rotation_keys,
        &mut scratch,
    )?;

    // 6. EvalMod, and the parts recombined into `Re + i·Im`.
    let tsk = keys.tensor_key();
    module.ckks_eval_mod(&mut out, &re, &ctx.eval_mod, tsk, &mut scratch)?;
    if let Some(double) = ctx.packed {
        // The second batch came out times `i`: one rotation by `k` values adds it.
        let k = 1i64 << double.log_k();
        module.ckks_rotate_into((k / 2) << double.log_r(), &mut re, &out, rotation_keys, &mut scratch)?;
        module.ckks_add_assign(&mut out, &re, &mut scratch)?;
        out.set_log_sparsity(GLWEPlaintextInfos::log_sparsity(&out) + 1);
    } else {
        module.ckks_eval_mod(&mut re, &im, &ctx.eval_mod, tsk, &mut scratch)?;
        module.ckks_mul_i_assign(&mut re, &mut scratch)?;
        module.ckks_add_assign(&mut out, &re, &mut scratch)?;
    }
    out.set_slots(SlotsKind::Complex);
    out.set_slot_layout(SlotLayout::Replicated(plan.layout));

    // 7. `Δ_b·m` at the raised scale is `m` at `Δ_b`, lifted by ModUp; one
    //    shaped rounding returns to the leveled scale.
    debug_assert_eq!(out.log_delta(), s_raised);
    out.set_scale(Scale::Log(plan.log_delta + plan.mod_up_extra_bits));
    dst.set_k((out.log_budget() + plan.level_log_delta).into());
    module.ckks_round_into(dst, &out, &mut scratch)
}

/// Shaping depths of the two ring maps of [`ckks_replicated_ci_bootstrap_into`]
/// for an input at scale `2^log_delta_in` and an output at the leveled scale.
fn ci_bootstrap_depths(layout: &ReplicatedLayout, log_delta_in: usize, level_log_delta: usize) -> (usize, usize) {
    let profile = crate::shaping::ShapingProfile::new(layout);
    (
        profile.shaping_depth(log_delta_in, layout.w()),
        profile.shaping_depth(level_log_delta, layout.w()),
    )
}

/// Scratch of [`ckks_replicated_ci_bootstrap_into`], `(scratch, scratch_ci)`:
/// the working ciphertexts of both rings beside the largest operation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_replicated_ci_bootstrap_tmp_bytes<BE, BCI, C, I, F, A, T, S, K>(
    module: &Module<BE>,
    ci: &Module<BCI>,
    ct_out: &C,
    ct_in: &I,
    ctx: &ReplicatedBootstrapping<BE, F>,
    atk_infos: &A,
    tsk_infos: &T,
    encapsulation_infos: Option<(&S, &S)>,
    conversion_infos: &K,
) -> (usize, usize)
where
    BE: Backend<Ring = Standard>,
    BCI: Backend<Ring = ConjugateInvariant>,
    Module<BE>: CKKSReplicatedBootstrapOps<BE> + CKKSCIRingMapOps<BE> + CKKSCopyOps<BE> + CKKSRoundOps<BE>,
    Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
    C: CKKSCtBounds,
    I: CKKSCtBounds,
    A: GGLWEInfos,
    T: GGLWEInfos,
    S: GGLWEInfos,
    K: GGLWEInfos,
{
    let plan = ctx.plan;
    let (lambda_in, lambda_out) = ci_bootstrap_depths(&plan.layout, ct_in.log_delta(), plan.level_log_delta);
    let ckks = |n: usize, k: usize, meta: GLWEPlaintextMeta| CKKSLayout {
        glwe_layout: GLWELayout {
            n: n.into(),
            k: k.into(),
            ..ct_in.glwe_layout()
        },
        meta,
    };
    let (n, n_ci) = (module.n(), ct_in.n().as_usize());
    let (k_in, k_out) = (ct_in.k().as_usize(), ct_out.k().as_usize() + 1);
    let level_meta = GLWEPlaintextMeta {
        scale: Scale::Log(plan.level_log_delta),
        ..ct_in.meta()
    };
    let embedded = ckks(n, k_in + lambda_in, ct_in.meta());
    let input = ckks(n, k_in, ct_in.meta());
    let output = ckks(n, k_out, level_meta);
    let wide_output = ckks(n, k_out + lambda_out, level_meta);
    let traced = ckks(n_ci, k_out + lambda_out - 1, level_meta);
    let ops = module
        .ckks_ci_embed_tmp_bytes(&embedded, conversion_infos)
        .max(module.ckks_round_tmp_bytes(&input, &embedded))
        .max(module.ckks_replicated_bootstrap_tmp_bytes(&output, &input, ctx, atk_infos, tsk_infos, encapsulation_infos))
        .max(module.ckks_copy_tmp_bytes(&wide_output, &output))
        .max(module.ckks_ci_trace_tmp_bytes(&traced, &wide_output, conversion_infos));
    let standard = [&embedded, &input, &output, &wide_output]
        .iter()
        .map(|layout| ct_bytes::<BE>(&layout.glwe_layout))
        .sum::<usize>()
        + ops;
    let wide_input = ckks(n_ci, k_in + lambda_in, ct_in.meta());
    let ci_ops = ci
        .ckks_copy_tmp_bytes(&wide_input, ct_in)
        .max(ci.ckks_round_tmp_bytes(ct_out, &traced));
    let conjugate_invariant = ct_bytes::<BCI>(&wide_input.glwe_layout) + ct_bytes::<BCI>(&traced.glwe_layout) + ci_ops;
    (standard, conjugate_invariant)
}

/// Bootstraps the conjugate-invariant replicated ciphertext `src` of degree
/// `N` into `dst` through the standard ring of degree `2N` of `module`, where
/// `ctx` was compiled: `src` is embedded, the standard bootstrapping refreshes
/// it, and the relative trace brings it back. Each ring map runs at the
/// shaping depth under the scale and rounds with shaping, so their key
/// switches and the trace's halving stay off the data slots. `dst` ends at
/// its own width and the leveled scale, which the standard bootstrapping
/// reaches one bit wider for the trace to spend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_replicated_ci_bootstrap_into<BE, BCI, Dst, Src, F, K, E, T>(
    module: &Module<BE>,
    ci: &Module<BCI>,
    dst: &mut Dst,
    src: &Src,
    ctx: &ReplicatedBootstrapping<BE, F>,
    workspace: &mut ReplicatedBootstrapWorkspace<BE>,
    keys: &K,
    embed_key: &GLWECIEmbedKeyPrepared<E, BE>,
    trace_key: &GLWECITraceKeyPrepared<T, BE>,
    scratch: &mut ScratchArena<'_, BE>,
    scratch_ci: &mut ScratchArena<'_, BCI>,
) -> Result<()>
where
    BE: Backend<Ring = Standard>,
    BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + for<'a> Backend<BufRef<'a> = BE::BufRef<'a>, BufMut<'a> = BE::BufMut<'a>>
        + 'static,
    Module<BE>: CKKSReplicatedBootstrapOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRoundOps<BE>
        + GLWECIEmbed<BE>
        + GLWECITrace<BE>
        + GLWEKeyswitch<BE>,
    Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
    Dst: GLWEToBackendMut<BCI> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BCI> + CKKSCtBounds,
    K: BootstrappingKeys<BE>,
    E: Data,
    T: Data,
    GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
    GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>,
{
    const OP: &str = "ckks_replicated_ci_bootstrap_into";
    check_destination(OP, dst, src)?;
    let plan = ctx.plan;
    if src.slot_layout() != SlotLayout::Replicated(plan.layout) {
        return Err(invalid(OP, "the source must have the bootstrapping's replicated layout"));
    }
    if module.n() != 2 * src.n().as_usize()
        || ci.n() != src.n().as_usize()
        || embed_key.n().as_usize() != module.n()
        || trace_key.n().as_usize() != module.n()
    {
        return Err(invalid(
            OP,
            "takes a standard module and conversion keys of degree 2N, a conjugate-invariant module and ciphertexts of degree N",
        ));
    }
    let (lambda_in, lambda_out) = ci_bootstrap_depths(&plan.layout, src.log_delta(), plan.level_log_delta);
    let standard = |k: usize| GLWELayout {
        n: module.n().into(),
        k: k.into(),
        ..src.glwe_layout()
    };
    let invariant = |k: usize| GLWELayout {
        k: k.into(),
        ..src.glwe_layout()
    };
    let (input, mut scratch) = scratch
        .borrow()
        .take_ckks_ciphertext_scratch(&standard(src.k().as_usize()), src.meta());
    let mut input = input;

    // 1. Into the standard ring, `lambda_in` bits under the scale.
    scratch.scope(|scratch| -> Result<()> {
        let (mut wide, mut scratch_ci) = scratch_ci
            .borrow()
            .take_ckks_ciphertext_scratch(&invariant(src.k().as_usize() + lambda_in), src.meta());
        ci.ckks_copy(&mut wide, src, &mut scratch_ci)?;
        wide.set_log_delta(src.log_delta() + lambda_in);
        let wide = wide.rebind::<BE>();
        let meta = GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..wide.meta()
        };
        let (mut embedded, mut scratch) = scratch.take_ckks_ciphertext_scratch(&standard(wide.k().as_usize()), meta);
        // `ckks_ci_embed` on the conjugate-invariant view carved from `scratch_ci`.
        module.glwe_ci_embed(&mut embedded, &wide);
        module.glwe_keyswitch_assign(&mut embedded, &embed_key.to_backend_ref(), &mut scratch);
        input.set_k((embedded.log_budget() + src.log_delta()).into());
        module.ckks_round_into(&mut input, &embedded, &mut scratch)
    })?;

    // 2. The standard bootstrapping, one bit wider than `dst`.
    let level_meta = GLWEPlaintextMeta {
        scale: Scale::Log(plan.level_log_delta),
        ..src.meta()
    };
    let k_out = dst.k().as_usize() + 1;
    let (mut output, mut scratch) = scratch.take_ckks_ciphertext_scratch(&standard(k_out), level_meta);
    module.ckks_replicated_bootstrap_into(&mut output, &input, ctx, workspace, keys, &mut scratch)?;

    // 3. Back by the relative trace, `lambda_out` bits under the scale.
    let (mut wide, mut scratch) = scratch.take_ckks_ciphertext_scratch(&standard(k_out + lambda_out), output.meta());
    module.ckks_copy(&mut wide, &output, &mut scratch)?;
    wide.set_log_delta(output.log_delta() + lambda_out);
    // `ckks_ci_trace` into the conjugate-invariant view carved from `scratch_ci`:
    // the trace doubles the real part, which one bit less of budget halves.
    let (mut switched, mut scratch) = scratch.take_ckks_ciphertext_scratch(&wide.glwe_layout(), wide.meta());
    module.glwe_keyswitch(&mut switched, &wide, &trace_key.to_backend_ref(), &mut scratch);
    let meta = GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..switched.meta()
    };
    let (traced, mut scratch_ci) = scratch_ci
        .borrow()
        .take_ckks_ciphertext_scratch(&invariant(k_out + lambda_out - 1), meta);
    let mut traced = traced.rebind::<BE>();
    traced.set_log_budget(switched.log_budget() - 1);
    module.glwe_ci_trace(&mut traced, &switched, &mut scratch);
    let traced = traced.rebind::<BCI>();
    dst.set_k((traced.log_budget() + plan.level_log_delta).into());
    ci.ckks_round_into(dst, &traced, &mut scratch_ci)
}
