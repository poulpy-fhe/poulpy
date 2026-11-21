use crate::CKKSResult as Result;
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::{
    Dnum, GGLWEInfos, GGLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision,
};
use poulpy_hal::layouts::{Module, Ring, ScratchArena};

use crate::oep::derived::round::{ckks_wide_copy, ckks_wide_copy_tmp_bytes, wide_k};
use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout,
    api::{CKKSCopyOps, CKKSInnerSumOps, CKKSLinearTransformationOps, CKKSRoundOps, ckks_plaintext_diagonal_layout},
    checked_log_budget_sub, ckks_ensure_operand,
    layouts::{
        CKKSCiphertextViewMut, ReplicatedConversion, ScratchArenaTakeCKKSCiphertext,
        replicated::ReplicatedChain,
        validation::{automorphism_key_layout, ensure_full_layout, ensure_replicated_operands, ensure_scratch},
    },
    oep::CKKSReplicatedImpl,
};

/// Scratch of both conversions of `ctx` from `a` into `res`, with keys of layout `key_infos`: the wide ciphertext and
/// the rounded one of a narrower destination, beside the largest of their steps, each factor with its baby steps.
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
    ckks_wide_copy_tmp_bytes::<BE, Rg, _>(module, a, ctx.extra_bits, a.log_budget(), |wide| {
        let diagonal = ckks_plaintext_diagonal_layout(wide);
        let factors = ctx.to.factors.iter().chain(ctx.from.iter().flat_map(|from| &from.factors));
        let ops = factors
            .map(|(_, plan)| {
                module.ckks_linear_transformation_eval_self_tmp_bytes(false, wide, &diagonal, plan, key_infos, key_infos)
            })
            .max()
            .unwrap_or(0)
            .max(module.ckks_copy_tmp_bytes(res, wide))
            .max(module.ckks_round_tmp_bytes(res, wide))
            .max(module.ckks_round_tmp_bytes(wide, wide))
            .max(module.ckks_inner_sum_tmp_bytes(wide, wide, key_infos));
        BE::scratch_aligned(module.glwe_bytes_of_from_infos(wide)) + ops
    })
}

/// `InsufficientHomomorphicCapacity` for `op` unless a result of `log_budget` bits at scale `2^log_delta` lands at
/// `res.k()`, a narrower `res` taking it through a modulus drop.
fn ensure_landing<R: CKKSCtBounds>(op: &'static str, res: &R, log_budget: usize, log_delta: usize) -> Result<()> {
    let drop = (log_budget + log_delta).saturating_sub(res.k().as_usize());
    checked_log_budget_sub(op, log_budget, drop)?;
    Ok(())
}

/// Both conversions once their operands and budget hold: every key of `chain` checked at the width it switches and the
/// arena against the selected query at those keys' layouts, then `M(a)` for the map `M` of `chain`, in place on a copy
/// of `a` relabelled `extra_bits` under its scale and on the full layout, off which the rotations of `M` take the data.
#[expect(clippy::too_many_arguments, reason = "the operands of a conversion and the chain it runs")]
fn wide_stage<'s, BE, Rg: Ring, R, A, H>(
    op: &'static str,
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &ReplicatedConversion<BE, Rg>,
    chain: &ReplicatedChain<BE, Rg>,
    keys: &H,
    scratch: &'s mut ScratchArena<'_, BE>,
) -> Result<(CKKSCiphertextViewMut<'s, BE, Rg>, ScratchArena<'s, BE>)>
where
    BE: CKKSReplicatedImpl<Rg> + 's,
    R: CKKSCtBounds,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
{
    let k = wide_k(a, ctx.extra_bits, a.log_budget());
    let mut bytes = None;
    for &(p, below) in &chain.keys {
        let key = automorphism_key_layout(op, keys, p, k - below)?;
        bytes = bytes.max(Some(BE::ckks_replicated_conversion_tmp_bytes(module, res, a, &key, ctx)));
    }
    // No switch runs without a key, so any key layout serves: one digit of the source's width.
    let bytes = bytes.unwrap_or_else(|| {
        let keyless = GGLWELayout {
            n: a.n(),
            base2k: a.base2k(),
            dnum: Dnum(1),
            k_aux: TorusPrecision(a.base2k().as_u32()),
            rank_in: a.rank(),
            rank_out: a.rank(),
            k_digit: a.k(),
            k_offset: TorusPrecision(0),
            stride: 1,
        };
        BE::ckks_replicated_conversion_tmp_bytes(module, res, a, &keyless, ctx)
    });
    ensure_scratch(op, scratch, bytes)?;
    let (mut x, mut rest) = ckks_wide_copy(module, a, ctx.extra_bits, a.log_budget(), scratch)?;
    x.set_slot_layout(SlotLayout::Full);
    for (lt, _) in &chain.factors {
        module.ckks_linear_transformation_eval_self_assign(&mut x, lt, keys, &mut rest)?;
    }
    Ok((x, rest))
}

pub(crate) fn ckks_to_replicated_derived<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &ReplicatedConversion<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSReplicatedImpl<Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
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
    let log_budget = checked_log_budget_sub(OP, a.log_budget(), ctx.to.consumed)?;
    ensure_landing(OP, res, log_budget, ctx.log_delta)?;
    let (mut x, mut scratch) = wide_stage(OP, module, &*res, a, ctx, &ctx.to, keys, scratch)?;
    x.set_slot_layout(SlotLayout::Replicated(ctx.layout));
    let k = log_budget + ctx.log_delta;
    if k <= res.k().as_usize() {
        res.set_k(k.into());
        return module.ckks_round(res, &x, &mut scratch);
    }
    // A narrower destination takes the rounded result through a modulus drop.
    let (mut rounded, mut scratch) = scratch.take_ckks_ciphertext_like_scratch::<_, Rg>(&x);
    rounded.set_log_delta(ctx.log_delta);
    module.ckks_round(&mut rounded, &x, &mut scratch)?;
    module.ckks_copy(res, &rounded, &mut scratch)
}

pub(crate) fn ckks_from_replicated_derived<BE, Rg: Ring, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &ReplicatedConversion<BE, Rg>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSReplicatedImpl<Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
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
    let levels = ctx.trace_levels;
    // The mean of the trace moves `levels` budget bits under the scale.
    let log_budget = checked_log_budget_sub(OP, a.log_budget(), from.consumed + levels)?;
    ensure_landing(OP, res, log_budget, a.log_delta() + ctx.extra_bits + levels)?;
    let (mut x, mut scratch) = wide_stage(OP, module, &*res, a, ctx, from, keys, scratch)?;
    module.ckks_inner_sum_assign(ctx.layout.log_k(), levels, true, &mut x, keys, &mut scratch)?;
    // The trace closes the slot cycle, which leaves a polynomial in `X^(2^levels)`.
    x.set_log_sparsity(levels);
    module.ckks_copy(res, &x, &mut scratch)
}
