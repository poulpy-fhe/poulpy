use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    Dnum, GGLWEInfos, GGLWELayout, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
    LWEInfos, LinearTransformationBabySteps, LinearTransformationPrepared, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSModuleInfos, ReplicatedLayout, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout,
    api::{CKKSAddOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSPow2Ops, CKKSRotateOps, CKKSRoundOps},
    checked_log_budget_sub, ckks_ensure_operand,
    layouts::{
        CKKSCiphertextViewMut, ReplicatedConversion, ReplicatedTransform, ReplicatedTransformWorkspace,
        ScratchArenaTakeCKKSCiphertext,
        validation::{
            ensure_full_layout, ensure_replicated_operands, ensure_rotation_keys, ensure_scratch, ensure_transform_workspace,
            transform_key_lookups,
        },
    },
    oep::CKKSReplicatedImpl,
    reference::linear_transformation::PlaintextDiagonalKeys,
};

/// The working layout of a stage on `ct`: `extra_bits` under its scale and
/// room for a trace's division by `2^trace_steps`.
pub(crate) fn working_layout<C: GLWEInfos>(ct: &C, extra_bits: usize, trace_steps: usize) -> GLWELayout {
    GLWELayout {
        k: (ct.k().as_usize() + extra_bits + trace_steps).into(),
        ..ct.glwe_layout()
    }
}

/// Wide stage onto a replicated layout: `res = M(a)` computed `extra_bits` under `a`'s scale, then rounded with
/// `layout`'s shaping to `log_delta`, landing at `res.k()` through a modulus drop when `res` is narrower. The caller
/// has checked the operands, workspace, keys and arena.
#[allow(clippy::too_many_arguments)]
fn wide_onto_replicated<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    transform: &ReplicatedTransform<BE, Rg>,
    workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
    layout: ReplicatedLayout,
    log_delta: usize,
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg> + CKKSCopyOps<BE, Rg> + CKKSRoundOps<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg>,
{
    let working = working_layout(a, extra_bits, 0);
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch::<_, Rg>(&working, a.meta());
    let (mut acc, mut scratch) = scratch.take_ckks_ciphertext_scratch::<_, Rg>(&working, a.meta());

    module.ckks_copy(&mut x, a, &mut scratch)?;
    x.set_log_delta(a.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_transform(module, &mut acc, &x, transform, workspace, keys, &mut scratch)?;
    acc.set_slot_layout(SlotLayout::Replicated(layout));
    let k = acc.log_budget() + log_delta;
    if k <= res.k().as_usize() {
        res.set_k(k.into());
        return module.ckks_round(res, &acc, &mut scratch);
    }
    x.set_meta(acc.meta());
    x.set_scale(crate::Scale::Log(log_delta));
    x.set_log_budget(acc.log_budget());
    module.ckks_round(&mut x, &acc, &mut scratch)?;
    module.ckks_copy(res, &x, &mut scratch)
}

/// Wide stage gathering onto period `k`: `res = T(M(a))` computed `extra_bits` under `a`'s scale, where the
/// normalized trace `T` sums the rotations by `trace` after dividing by `2^trace.len()`, which lifts the modulus so
/// that its multiples vanish. Rounds plainly, and lands at `res.k()`.
#[allow(clippy::too_many_arguments)]
fn wide_gather_trace<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    transform: &ReplicatedTransform<BE, Rg>,
    workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
    trace: &[i64],
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg>
        + CKKSRotateOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg>,
{
    let working = working_layout(a, extra_bits, trace.len());
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch::<_, Rg>(&working, a.meta());
    let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch::<_, Rg>(&working, a.meta());
    let (mut rotated, mut scratch) = scratch.take_ckks_ciphertext_scratch::<_, Rg>(&working, a.meta());

    module.ckks_copy(&mut x, a, &mut scratch)?;
    x.set_log_delta(a.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_transform(module, &mut acc, &x, transform, workspace, keys, &mut scratch)?;
    normalized_trace(module, &mut acc, trace, keys, &mut rotated, &mut scratch)?;
    module.ckks_copy(res, &acc, &mut scratch)
}

/// `ct = (1/2^L) Σ_(i < 2^L) rot_(i·step)(ct)` by `L` rotations by `step, 2·step, ...`.
/// The division comes first and lifts the modulus: the trace of the modulus'
/// multiples lies in `2^L·R`, so they vanish on the torus. A trace closing the
/// slot cycle, `step·2^L = N/2`, leaves a polynomial in `X^(2^L)`.
fn normalized_trace<BE, Rg: Ring, H>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextViewMut<'_, BE, Rg>,
    shifts: &[i64],
    keys: &H,
    rotated: &mut CKKSCiphertextViewMut<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE, Rg> + CKKSAddOps<BE, Rg> + CKKSPow2Ops<BE, Rg>,
    H: GetAutomorphismKey<BE, Rg>,
{
    module.ckks_div_pow2_assign(shifts.len(), ct)?;
    for &shift in shifts {
        module.ckks_rotate_into(shift, rotated, &*ct, keys, scratch)?;
        module.ckks_add_assign(ct, &*rotated, scratch)?;
    }
    if shifts
        .first()
        .is_some_and(|&step| (step as usize) << shifts.len() == module.ckks_slots_of_degree::<Rg>(ct.n().as_usize()))
    {
        ct.set_log_sparsity(shifts.len());
    }
    Ok(())
}

/// `acc = M(x)` by the prepared transform `M`, its factors chained through one temporary at the layout of `x`, so
/// that the last lands in `acc`. Its diagonals are plaintexts, so it reads the automorphism keys of `keys` only.
fn apply_transform<BE, Rg: Ring, H>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertextViewMut<'_, BE, Rg>,
    x: &CKKSCiphertextViewMut<'_, BE, Rg>,
    transform: &ReplicatedTransform<BE, Rg>,
    workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg>,
    H: GetAutomorphismKey<BE, Rg>,
{
    let keys = &PlaintextDiagonalKeys(keys);
    let count = transform.factors.len();
    let (mut tmp, mut scratch) = scratch
        .borrow()
        .take_ckks_ciphertext_scratch::<_, Rg>(&x.glwe_layout(), x.meta());
    for (i, ((lt, _), babies)) in transform.factors.iter().zip(workspace.babies.iter_mut()).enumerate() {
        // The outputs alternate between `acc` and `tmp`, the last in `acc`.
        match (i, (count - 1 - i).is_multiple_of(2)) {
            (0, true) => eval_factor(module, acc, x, lt, babies, keys, &mut scratch)?,
            (0, false) => eval_factor(module, &mut tmp, x, lt, babies, keys, &mut scratch)?,
            (_, true) => eval_factor(module, acc, &tmp, lt, babies, keys, &mut scratch)?,
            (_, false) => eval_factor(module, &mut tmp, &*acc, lt, babies, keys, &mut scratch)?,
        }
    }
    Ok(())
}

/// `out = M(input)` for one factor `M`, its baby steps prepared from `input`.
fn eval_factor<BE, Rg: Ring, H, Out, In>(
    module: &Module<BE>,
    out: &mut Out,
    input: &In,
    lt: &LinearTransformationPrepared<BE, Rg>,
    babies: &mut LinearTransformationBabySteps<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg>,
    H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    Out: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    In: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
{
    module.ckks_linear_transformation_baby_steps_prepare(babies, input, keys, scratch)?;
    module.ckks_linear_transformation_eval_into(out, input, babies, lt, keys, scratch)
}

/// Scratch of [`apply_transform`] on inputs of layout `wide`: the temporary of
/// a chain beside the largest factor, each at its input width.
fn transform_tmp_bytes<BE, Rg: Ring, K>(
    module: &Module<BE>,
    wide: &CKKSLayout,
    key_infos: &K,
    transform: &ReplicatedTransform<BE, Rg>,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg>,
    K: GGLWEInfos,
{
    let factors = transform
        .factor_inputs(&wide.glwe_layout)
        .into_iter()
        .map(|glwe_layout| {
            factor_tmp_bytes(
                module,
                &CKKSLayout {
                    glwe_layout,
                    meta: wide.meta,
                },
                key_infos,
            )
        })
        .max()
        .unwrap_or(0);
    ct_bytes::<BE>(&wide.glwe_layout) + factors
}

fn factor_tmp_bytes<BE, Rg: Ring, K>(module: &Module<BE>, wide: &CKKSLayout, key_infos: &K) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE, Rg>,
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

/// Scratch of both conversions of `ctx` from `a` into `res`, with keys of layout `key_infos`: the three working
/// ciphertexts of a wide stage beside the largest of its steps.
pub(crate) fn ckks_replicated_conversion_tmp_bytes_derived<BE, Rg: Ring, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    key_infos: &K,
    ctx: &ReplicatedConversion<BE, Rg>,
) -> usize
where
    BE: CKKSReplicatedImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    K: GGLWEInfos,
{
    let layout = working_layout(a, ctx.extra_bits, ctx.trace.len());
    let wide = CKKSLayout {
        glwe_layout: layout,
        meta: a.meta(),
    };
    let ops = transform_tmp_bytes(module, &wide, key_infos, &ctx.to)
        .max(
            ctx.from
                .as_ref()
                .map_or(0, |from| transform_tmp_bytes(module, &wide, key_infos, from)),
        )
        .max(module.ckks_rotate_tmp_bytes(&wide, key_infos))
        .max(module.ckks_add_tmp_bytes(layout.size()))
        .max(module.ckks_copy_tmp_bytes(&wide, a))
        .max(module.ckks_copy_tmp_bytes(res, &wide))
        .max(module.ckks_round_tmp_bytes(res, &wide))
        .max(module.ckks_round_tmp_bytes(&wide, &wide))
        .max(module.ckks_div_pow2_tmp_bytes(layout.size()));
    3 * ct_bytes::<BE>(&layout) + ops
}

/// The selected query of the conversions over the layouts of the keys a run looks up, or of a keyless switch when it
/// looks none up.
fn conversion_bytes<BE, Rg: Ring, R, A>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    keys: &[GGLWELayout],
    ctx: &ReplicatedConversion<BE, Rg>,
) -> usize
where
    BE: CKKSReplicatedImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
{
    let query = |key: &GGLWELayout| BE::ckks_replicated_conversion_tmp_bytes(module, res, a, key, ctx);
    keys.iter().map(query).max().unwrap_or_else(|| {
        // No switch runs, so any key layout serves: one digit of the source's width.
        query(&GGLWELayout {
            n: a.n(),
            base2k: a.base2k(),
            dnum: Dnum(1),
            k_aux: TorusPrecision(a.base2k().as_u32()),
            rank_in: a.rank(),
            rank_out: a.rank(),
            k_digit: a.k(),
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    })
}

/// The rotations, and the widths, at which `ckks_to_replicated` of `a` through `ctx` looks its keys up: the width
/// model of its entry check.
pub(crate) fn to_key_lookups<BE: Backend, Rg: Ring, A: GLWEInfos>(
    ctx: &ReplicatedConversion<BE, Rg>,
    workspace: &ReplicatedTransformWorkspace<BE, Rg>,
    a: &A,
) -> Vec<(i64, usize)> {
    transform_key_lookups(&ctx.to, workspace, working_layout(a, ctx.extra_bits, 0).k.as_usize())
}

/// The rotations, and the widths, at which `ckks_from_replicated` of `a` through `ctx` looks its keys up: the gather
/// at the working width, then the trace at the width it leaves, a result of `log_budget` bits.
pub(crate) fn from_key_lookups<BE: Backend, Rg: Ring, A: CKKSCtBounds>(
    ctx: &ReplicatedConversion<BE, Rg>,
    from: &ReplicatedTransform<BE, Rg>,
    workspace: &ReplicatedTransformWorkspace<BE, Rg>,
    a: &A,
    log_budget: usize,
) -> Vec<(i64, usize)> {
    let trace_k = log_budget + a.log_delta() + ctx.extra_bits + ctx.trace.len();
    transform_key_lookups(from, workspace, working_layout(a, ctx.extra_bits, 0).k.as_usize())
        .into_iter()
        .chain(ctx.trace.iter().map(|&rotation| (rotation, trace_k)))
        .collect()
}

/// `InsufficientHomomorphicCapacity` for `op` unless a result of `log_budget` bits at scale `2^log_delta` lands at
/// `res.k()`, a narrower `res` taking it through a modulus drop.
fn ensure_landing<R: CKKSCtBounds>(op: &'static str, res: &R, log_budget: usize, log_delta: usize) -> Result<()> {
    let drop = (log_budget + log_delta).saturating_sub(res.k().as_usize());
    checked_log_budget_sub(op, log_budget, drop)?;
    Ok(())
}

pub(crate) fn ckks_to_replicated_derived<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &ReplicatedConversion<BE, Rg>,
    workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSReplicatedImpl<Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg>,
{
    const OP: &str = "ckks_to_replicated";
    ensure_replicated_operands(OP, module.n(), ctx.n.as_usize(), a, res)?;
    ensure_full_layout(OP, a.slot_layout())?;
    ckks_ensure_operand!(
        OP,
        ctx.log_delta <= a.log_delta() + ctx.extra_bits,
        "the replicated scale 2^{} exceeds the working scale 2^{} of the source",
        ctx.log_delta,
        a.log_delta() + ctx.extra_bits
    );
    let working = working_layout(a, ctx.extra_bits, 0);
    ensure_transform_workspace(OP, workspace, &working, &ctx.to)?;
    let log_budget = checked_log_budget_sub(OP, a.log_budget(), ctx.to.consumed_bits())?;
    ensure_landing(OP, res, log_budget, ctx.log_delta)?;
    let key_layouts = ensure_rotation_keys(OP, module, a.n().as_usize(), keys, to_key_lookups(ctx, workspace, a))?;
    ensure_scratch(OP, scratch, conversion_bytes(module, res, a, &key_layouts, ctx))?;
    wide_onto_replicated(
        module,
        res,
        a,
        &ctx.to,
        workspace,
        ctx.layout,
        ctx.log_delta,
        ctx.extra_bits,
        keys,
        scratch,
    )
}

pub(crate) fn ckks_from_replicated_derived<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &ReplicatedConversion<BE, Rg>,
    workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSReplicatedImpl<Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg>,
{
    const OP: &str = "ckks_from_replicated";
    ensure_replicated_operands(OP, module.n(), ctx.n.as_usize(), a, res)?;
    ckks_ensure_operand!(
        OP,
        a.slot_layout() == SlotLayout::Replicated(ctx.layout),
        "the source must have the conversion's replicated layout"
    );
    let Some(from) = &ctx.from else {
        return Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: "the gather needs the period to divide the batch size".into(),
        }
        .into());
    };
    let working = working_layout(a, ctx.extra_bits, 0);
    ensure_transform_workspace(OP, workspace, &working, from)?;
    // The trace divides by `2^trace.len()` first, moving that many budget bits under the scale.
    let log_budget = checked_log_budget_sub(OP, a.log_budget(), from.consumed_bits() + ctx.trace.len())?;
    let log_delta = a.log_delta() + ctx.extra_bits + ctx.trace.len();
    ensure_landing(OP, res, log_budget, log_delta)?;
    let lookups = from_key_lookups(ctx, from, workspace, a, log_budget);
    let key_layouts = ensure_rotation_keys(OP, module, a.n().as_usize(), keys, lookups)?;
    ensure_scratch(OP, scratch, conversion_bytes(module, res, a, &key_layouts, ctx))?;
    wide_gather_trace(module, res, a, from, workspace, &ctx.trace, ctx.extra_bits, keys, scratch)
}
