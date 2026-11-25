//! Validation shared across modules: generic key and operand layouts, and the refusals of the bootstrap steps and of the
//! bodies composing them.
//!
//! The generic helpers check the structural invariants (degree, rank, radix, decomposition shape, storage capacity, backend-view consistency) that any gadget key or runtime ciphertext operand must satisfy before entering the arithmetic primitives, whichever pipeline consumes them, and the uniformity of a linear transformation's diagonals.
//! The bootstrap refusals sit here because `oep::derived` and `reference` both read them, and `oep` names no `reference`; a check only one `reference` module reads, as PaCo's plan-against-material checks, stays there.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    LinearTransformationPrepared,
    layouts::{
        Base2K, GGLWEInfos, GGLWELayout, GLWEInfos, GLWELayout, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos,
        LWEInfos, LinearTransformation, TorusPrecision, ensure_key_covers, galois_is_identity,
        prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, Ring, ScratchArena, Standard, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSError, CKKSInfos, CKKSLayout, CKKSResult, GLWEPlaintextInfos, GLWEPlaintextMeta,
    Scale, SlotLayout, SlotsKind,
    api::{CKKSBootstrappingOps, LtDiagonalMeta},
    checked_log_budget_sub, ckks_bail, ckks_ensure, ckks_ensure_operand, ckks_log_delta,
    error::ensure_plaintext_operand,
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingLinearKeys, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSDFTFactors, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared, DFTPlan, DFTType,
        EncodedLut, EvalMod, EvalModPlan, EvalModType,
        dft::{charge_factor_keys, charge_key, trace_galois_element},
        slots::{OneCoeffs, one_coeffs},
    },
    presets::first_factor::{first_factor_small_bits, first_factor_work},
};

/// `InvalidOperand` for `op` unless `prepared` was allocated for `lt`'s layout: the same schedule, every diagonal of
/// both at one degree, a diagonal of `lt` raised to the backend's minimum degree, radix, rank, width and prepared limb
/// count (a ciphertext diagonal's live limbs, a plaintext's allocated ones), and `lt`'s baby steps led by the identity
/// rotation.
pub(crate) fn ensure_prepared_transformation_layout<BE: Backend, P: GLWEInfos, Rg: Ring>(
    op: &'static str,
    prepared: &LinearTransformationPrepared<BE, Rg>,
    lt: &LinearTransformation<P>,
) -> crate::CKKSResult<()> {
    let invalid = |reason: &str| -> crate::CKKSError {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: reason.into(),
        }
        .into()
    };
    if prepared.index() != lt.index() {
        return Err(invalid("the transformation's schedule differs from the prepared one's"));
    }
    let Some(first) = prepared.first_diagonal_plaintext() else {
        return Err(invalid("the prepared transformation has no diagonal"));
    };
    let layout = (first.glwe_layout(), first.max_size());
    let prepared_layouts = prepared
        .giant_steps
        .iter()
        .flat_map(|gs| &gs.diagonals)
        .map(|d| (d.plaintext.glwe_layout(), d.plaintext.max_size()));
    // A ciphertext diagonal is prepared at its width, a plaintext one across its allocation.
    let source_layouts = lt.giant_steps.iter().flat_map(|gs| &gs.diagonals).map(|d| {
        let p = &d.plaintext;
        let layout = GLWELayout {
            n: p.n().max(BE::MIN_DEGREE.into()),
            ..p.glwe_layout()
        };
        (layout, if p.rank().as_usize() > 0 { p.size() } else { p.max_size() })
    });
    if prepared_layouts.chain(source_layouts).any(|diagonal| diagonal != layout) {
        return Err(invalid("a diagonal's layout differs from the prepared transformation's"));
    }
    if lt.baby_galois.first().is_some_and(|&baby| !galois_is_identity(baby)) {
        return Err(invalid("the baby steps do not start with the identity"));
    }
    Ok(())
}

/// `InvalidOperand` for `op` unless every diagonal of `lt` shares the first one's scale, storage width and slot kind,
/// the `(log_scale, max_k, slots)` that `meta` reads.
///
/// The evaluation derives one `cnv_offset` and the result metadata from the first diagonal alone, and the preparation
/// stashes the first diagonal's scale and slot kind for all, so a hand-built transform with another scale on a later
/// diagonal would mis-scale its contribution and one with another slot kind would mislabel the result.
pub(crate) fn ensure_uniform_diagonals<P>(
    op: &'static str,
    lt: &LinearTransformation<P>,
    meta: impl Fn(&P) -> (usize, usize, SlotsKind),
) -> crate::CKKSResult<()> {
    let Some(first) = lt.first_diagonal_plaintext() else {
        return Ok(());
    };
    let (log_scale, max_k, slots) = meta(first);
    for gs in &lt.giant_steps {
        for diag in &gs.diagonals {
            let (diag_log_scale, diag_max_k, diag_slots) = meta(&diag.plaintext);
            if (diag_log_scale, diag_max_k, diag_slots) != (log_scale, max_k, slots) {
                return Err(CKKSCompositionError::InvalidOperand {
                    op,
                    reason: format!(
                        "linear transformation diagonals are not uniform: diagonal (giant Galois element {}, baby {}) has (log_scale {diag_log_scale}, max_k {diag_max_k}, {diag_slots:?}) but the first diagonal has ({log_scale}, {max_k}, {slots:?})",
                        gs.galois, diag.baby_galois,
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Checks degree/rank/radix compatibility shared by all rank-1 gadget keys.
pub(crate) fn validate_gadget_key<K: GGLWEInfos + ?Sized>(
    name: &str,
    key: &K,
    n: usize,
    base2k: Base2K,
    working_size: usize,
) -> Result<()> {
    ensure!(
        key.n().as_usize() == n,
        "{name} degree {} does not match the expected degree {n}",
        key.n()
    );
    ensure!(
        key.rank_in().as_usize() == 1 && key.rank_out().as_usize() == 1,
        "{name} must have rank_in=rank_out=1, got rank_in={} and rank_out={}",
        key.rank_in(),
        key.rank_out(),
    );
    ensure!(
        key.base2k() == base2k,
        "{name} base2k {} does not match the expected base2k {base2k}",
        key.base2k(),
    );
    ensure!(key.dnum().as_usize() > 0, "{name} must have at least one decomposition digit");
    ensure!(key.k_digit().as_usize() > 0, "{name} decomposition digits must be non-empty");
    ensure!(
        key.k().as_usize() > key.k_digit().as_usize(),
        "{name} decomposition digit of {} bits must be narrower than the {}-bit key",
        key.k_digit(),
        key.k(),
    );
    let gadget_k = key
        .dnum()
        .as_usize()
        .checked_mul(key.k_digit().as_usize())
        .with_context(|| format!("{name} decomposition size overflows usize"))?;
    ensure!(
        gadget_k <= key.k().as_usize(),
        "{name} decomposition covers {gadget_k} bits, exceeding its {}-bit key",
        key.k(),
    );
    let required_size = working_size
        .checked_add(key.dsize().as_usize())
        .with_context(|| format!("{name} required working width overflows usize"))?;
    ensure!(
        key.max_size() >= required_size,
        "{name} stores {} limbs, but operations require at least {required_size} ({working_size} working + {} guard)",
        key.max_size(),
        key.dsize(),
    );
    validate_storage_capacity(name, key)
}

/// Validates a custom key manager's reported gadget-key metadata against the
/// backend-native view that the arithmetic primitives actually consume.
///
/// Built-in prepared key wrappers forward these fields directly. Checking
/// both representations keeps safe key-manager extension points (e.g.
/// [`PaCoKeys`](crate::layouts::PaCoKeys)) from turning inconsistent
/// lazy-store metadata into assertion failures below the CKKS composition
/// layer.
pub(crate) fn validate_gadget_backend_view<R, V>(
    name: &str,
    reported: &R,
    view: &V,
    n: usize,
    base2k: Base2K,
    working_size: usize,
) -> Result<()>
where
    R: GGLWEInfos + ?Sized,
    V: GGLWEInfos + ?Sized,
{
    validate_gadget_key(name, reported, n, base2k, working_size)?;
    let view_name = format!("{name} backend view");
    validate_gadget_key(&view_name, view, n, base2k, working_size)?;
    ensure!(
        reported.n() == view.n()
            && reported.base2k() == view.base2k()
            && reported.k() == view.k()
            && reported.max_size() == view.max_size()
            && reported.rank_in() == view.rank_in()
            && reported.rank_out() == view.rank_out()
            && reported.dnum() == view.dnum()
            && reported.k_digit() == view.k_digit(),
        "{name} reported layout does not match its backend-native view",
    );
    Ok(())
}

/// Rejects metadata that claims more torus precision than the backing
/// allocation can hold. Active-limb sufficiency (a backend view truncated
/// below the metadata) is checked separately by
/// [`validate_backend_storage_capacity`].
pub(crate) fn validate_storage_capacity<K: LWEInfos + ?Sized>(name: &str, key: &K) -> Result<()> {
    let base2k = key.base2k().as_usize();
    ensure!((1..=63).contains(&base2k), "{name} base2k must be in [1, 63], got {base2k}");
    let capacity = key
        .max_size()
        .checked_mul(base2k)
        .with_context(|| format!("{name} storage capacity overflows usize"))?;
    ensure!(
        key.k().as_usize() <= capacity,
        "{name} torus width {} exceeds its {capacity}-bit storage capacity",
        key.k(),
    );
    Ok(())
}

/// Validates both allocated capacity and active backend limbs for a runtime
/// ciphertext-like operand. This catches truncated or malformed custom key
/// stores whose metadata still advertises more precision than their active
/// view exposes.
pub(crate) fn validate_backend_storage_capacity<BE, K, Rg: Ring>(name: &str, key: &K) -> Result<()>
where
    BE: Backend,
    K: GLWEInfos + GLWEToBackendRef<BE, Rg>,
{
    validate_storage_capacity(name, key)?;
    let view = key.to_backend_ref();
    ensure!(
        key.n() == view.n()
            && key.base2k() == view.base2k()
            && key.k() == view.k()
            && key.max_size() == view.max_size()
            && key.rank() == view.rank(),
        "{name} reported layout does not match its backend-native view",
    );
    let active = view.data().size();
    ensure!(
        active >= key.size(),
        "{name} exposes {active} active limbs, but its torus width requires {}",
        key.size(),
    );
    Ok(())
}

/// The common slot layout of a product's operands.
pub(crate) fn product_layout(op: &'static str, layouts: &[SlotLayout]) -> crate::CKKSResult<SlotLayout> {
    let first = layouts[0];
    if layouts.iter().any(|&layout| layout != first) {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "the operands' slot layouts differ".into(),
        }
        .into());
    }
    Ok(first)
}

/// The slot layout `pt` multiplies `a` in: that of `a` when `pt` is a scalar, a plaintext of degree 1 on the full
/// layout, which multiplies every slot of any layout, guards included at zero.
pub(crate) fn factor_layout<A: GLWEPlaintextInfos, P: GLWEPlaintextInfos + LWEInfos>(a: &A, pt: &P) -> SlotLayout {
    match pt.slot_layout() {
        SlotLayout::Full if pt.n().as_usize() == 1 => a.slot_layout(),
        layout => layout,
    }
}

/// Rejects a replicated slot layout in an operation that rounds plainly or
/// writes the guard slots.
pub(crate) fn ensure_full_layout(op: &'static str, slot_layout: SlotLayout) -> crate::CKKSResult<()> {
    if let SlotLayout::Replicated(_) = slot_layout {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "the operation does not support a replicated slot layout".into(),
        }
        .into());
    }
    Ok(())
}

/// `InvalidOperand` for `op` when one of the lists of constants `steps` is empty or holds a constant encoded for another
/// slot layout than the input's, `slot_layout`, and `PlaintextDegreeMismatch` for a constant whose degree does not embed
/// into the input's, `n`.
pub(crate) fn ensure_masked_constants<'a, P: GLWEPlaintextInfos + LWEInfos + 'a>(
    op: &'static str,
    slot_layout: SlotLayout,
    n: usize,
    steps: impl IntoIterator<Item = &'a [P]>,
) -> crate::CKKSResult<()> {
    for step in steps {
        ckks_ensure_operand!(op, !step.is_empty(), "a baby step holds no constant");
        ckks_ensure_operand!(
            op,
            step.iter().all(|c| c.slot_layout() == slot_layout),
            "the constants are encoded for another slot layout than the input's"
        );
        for c in step {
            crate::error::ensure_plaintext_degree_embeds(op, n, c.n().as_usize())?;
        }
    }
    Ok(())
}

/// `Internal` for `op` when `scratch` holds less than `bytes`, the body's query, checked before its first carve.
pub(crate) fn ensure_scratch<BE: Backend>(
    op: &'static str,
    scratch: &ScratchArena<'_, BE>,
    bytes: usize,
) -> crate::CKKSResult<()> {
    let available = scratch.available();
    ckks_ensure!(
        available >= bytes,
        "{op} needs a scratch arena of {bytes} bytes, got {available}"
    );
    Ok(())
}

/// Rejects a rotation by `k` slots that moves guard slots onto data slots.
pub(crate) fn ensure_admissible_rotation(op: &'static str, slot_layout: SlotLayout, k: i64) -> crate::CKKSResult<()> {
    if let SlotLayout::Replicated(layout) = slot_layout
        && k.rem_euclid(1 << layout.log_r()) != 0
    {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: format!("a rotation by {k} slots is not a multiple of the period 2^{}", layout.log_r()),
        }
        .into());
    }
    Ok(())
}

/// Rejects a sum of a replicated layout at two scales, which would drop the
/// finer operand's extra bits with plain rounding.
pub(crate) fn reject_replicated_scales(
    op: &'static str,
    slot_layout: crate::SlotLayout,
    log_delta_a: usize,
    log_delta_b: usize,
) -> crate::CKKSResult<()> {
    if matches!(slot_layout, crate::SlotLayout::Replicated(_)) && log_delta_a != log_delta_b {
        return Err(crate::CKKSCompositionError::InvalidOperand {
            op,
            reason: format!(
                "a replicated layout adds at one scale, got 2^{log_delta_a} and 2^{log_delta_b}: round the finer operand with shaping first"
            ),
        }
        .into());
    }
    Ok(())
}

/// The refusals of the plaintext route of `ckks_add_assign` and `ckks_sub_assign` of `pt` into `dst` under `op`, before
/// its plaintext operand checks: `dst` above the module's degree, unequal slot layouts and, on a replicated layout, a
/// plaintext finer than `dst`.
pub(crate) fn check_add_pt<M: ModuleN + ?Sized, D: CKKSInfos, P: CKKSInfos>(
    op: &'static str,
    module: &M,
    dst: &D,
    pt: &P,
) -> CKKSResult<()> {
    crate::ensure_operand_degrees(op, module, [dst.n()])?;
    let slot_layout = product_layout(op, &[dst.slot_layout(), pt.slot_layout()])?;
    if pt.log_delta() > dst.log_delta() {
        reject_replicated_scales(op, slot_layout, dst.log_delta(), pt.log_delta())?;
    }
    Ok(())
}

/// `InvalidOperand` for `op` unless `a` and `res` have rank 1: the steps carve rank-1 working ciphertexts, which a
/// higher rank would silently truncate.
pub(crate) fn ensure_rank_one<A: GLWEInfos, R: GLWEInfos>(op: &'static str, a: &A, res: &R) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        a.rank().as_usize() == 1 && res.rank().as_usize() == 1,
        "supports rank-1 ciphertexts only, got rank {} -> {}",
        a.rank().as_usize(),
        res.rank().as_usize()
    );
    Ok(())
}

/// `InvalidOperand` for `op` unless `a` and `res` share a degree that the module of degree `module_n` serves.
pub(crate) fn ensure_one_degree<A: GLWEInfos, R: GLWEInfos>(op: &'static str, module_n: usize, a: &A, res: &R) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        res.n() == a.n() && a.n().as_usize() <= module_n,
        "takes ciphertexts of one degree, which the module serves"
    );
    Ok(())
}

/// `InvalidOperand` for `op` unless `a` and `res` share a base2k.
pub(crate) fn ensure_one_base2k<A: GLWEInfos, R: GLWEInfos>(op: &'static str, a: &A, res: &R) -> CKKSResult<()> {
    ckks_ensure_operand!(op, a.base2k() == res.base2k(), "takes ciphertexts of one base2k");
    Ok(())
}

/// `InvalidOperand` for `op` unless `a` and `res` share a rank and the degree `n` a replicated context is compiled at,
/// which the module of degree `module_n` serves.
pub(crate) fn ensure_replicated_operands<A: GLWEInfos, R: GLWEInfos>(
    op: &'static str,
    module_n: usize,
    n: usize,
    a: &A,
    res: &R,
) -> CKKSResult<()> {
    ensure_one_degree(op, module_n, a, res)?;
    ckks_ensure_operand!(
        op,
        a.n().as_usize() == n,
        "takes ciphertexts of degree {n}, the degree of its context, got {}",
        a.n().as_usize()
    );
    ckks_ensure_operand!(
        op,
        a.rank() == res.rank(),
        "takes a source and a destination of one rank, got {} and {}",
        a.rank().as_usize(),
        res.rank().as_usize()
    );
    Ok(())
}

/// The key of Galois element `p` at width `k`, its layout, or `MissingAutomorphismKey` or `AutomorphismKeyTooNarrow` for `op` naming `p`.
pub(crate) fn automorphism_key_layout<BE, Rg, H>(op: &'static str, keys: &H, p: i64, k: usize) -> CKKSResult<GGLWELayout>
where
    BE: Backend,
    Rg: Ring,
    H: GetAutomorphismKey<BE, Rg>,
{
    let k = TorusPrecision(k as u32);
    keys.get_automorphism_key(p, k)
        .map(|key| key.gglwe_layout())
        .map_err(|source| CKKSCompositionError::automorphism_key_lookup(op, p, k.into(), source).into())
}

/// `InvalidOperand` for `op` unless `keys` carry the encapsulation switches exactly when the recipe encapsulates.
pub(crate) fn ensure_encapsulation<BE, K>(
    op: &'static str,
    keys: &K,
    sparse_secret_hamming_weight: Option<usize>,
) -> CKKSResult<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
{
    ckks_ensure_operand!(
        op,
        keys.encapsulation_keys().is_some() == sparse_secret_hamming_weight.is_some(),
        "bootstrapping key encapsulation does not match the compiled recipe"
    );
    Ok(())
}

/// `InvalidOperand` for an input of another degree than the transforms of `ctx` fill, a plan on fewer slots than the
/// ring taking the gap that keeps the overflow of the raise in the subring they read, or for an input not labelled in
/// that subring, whose other coefficients the raise drops.
fn ensure_bootstrap_degree<BE: Backend, F>(
    op: &'static str,
    ctx: &BootstrappingContextPrepared<BE, F>,
    a: &CKKSCiphertextOwned<BE>,
) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        ctx.bootstrap_degree() == a.n().as_usize(),
        "the transforms of the context must fill the slots of the input's degree, or of the subring of its gap"
    );
    ckks_ensure_operand!(
        op,
        crate::GLWEPlaintextInfos::log_sparsity(a) >= ctx.log_gap(),
        "the input must lie in the subring of the gap of the context"
    );
    Ok(())
}

/// What the chain of `dft` on a replicated layout refuses before its trace, on a working ciphertext of degree `n`, width
/// `k` and budget `log_budget`: a layout past the slots of degree `n`, less budget than its factors consume, and a key
/// of the trace or of a factor's schedule missing at its width. Returns the largest `key_bytes` of those keys, 0 on the
/// full layout.
pub(crate) fn check_replicated_chain<BE, Dir, Fmt, D, H>(
    op: &'static str,
    dft: &D,
    (n, k, log_budget): (usize, usize, usize),
    keys: &H,
    key_bytes: impl Fn(&GGLWELayout) -> usize,
) -> CKKSResult<usize>
where
    BE: Backend,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
    H: GetAutomorphismKey<BE>,
{
    let plan = dft.dft_plan();
    let SlotLayout::Replicated(layout) = plan.slot_layout() else {
        return Ok(0);
    };
    let log_n = n.ilog2() as usize;
    ckks_ensure_operand!(
        op,
        layout.log_k() + layout.log_r() < log_n,
        "the slots of the input's degree must hold the layout of the plan"
    );
    let scales = || {
        dft.factors()
            .map(|f| f.first_diagonal_plaintext().map_or(0, |pt| pt.lt_log_scale()))
    };
    checked_log_budget_sub(op, log_budget, scales().sum())?;
    let order = Standard::CYCLOTOMIC_ORDER_FACTOR * n as i64;
    let (mut offsets, mut below) = (BTreeMap::new(), 0);
    for level in plan.trace_levels(log_n).into_iter().flatten() {
        charge_key(&mut offsets, trace_galois_element(level, order), 0);
    }
    for (schedule, scale) in dft.factor_plans().iter().zip(scales()) {
        charge_factor_keys(&mut offsets, schedule, below, scale, order);
        below += scale;
    }
    offsets.remove(&galois_element(0, order));
    let mut bytes = 0;
    for (p, below) in offsets {
        bytes = bytes.max(key_bytes(&automorphism_key_layout(op, keys, p, k - below)?));
    }
    Ok(bytes)
}

/// `InvalidOperand` for `op` unless `slot_layout` is the layout the chain of `plan` reads: a `Decode` on a replicated
/// layout reads that layout, every other chain the full layout.
pub(crate) fn ensure_chain_input(op: &'static str, plan: &DFTPlan, slot_layout: SlotLayout) -> CKKSResult<()> {
    if plan.kind() == DFTType::Decode && plan.slot_layout() != SlotLayout::Full {
        ckks_ensure_operand!(
            op,
            slot_layout == plan.slot_layout(),
            "the source must have the replicated layout of SlotsToCoeffs"
        );
        return Ok(());
    }
    ensure_full_layout(op, slot_layout)
}

/// What `ckks_bootstrap_derived` refuses before its first step, so that no later step refuses with `res` half written:
/// the context kind, ranks, degrees and radices, ciphertexts of another degree than the one the context bootstraps, a
/// sparse plan without its gap among them, an input off the subring of the gap, or a module of another degree than
/// `2N` with EvalMod on the
/// conjugate-invariant ring, an input the lifts of a C2S-first raise do not fit, a working width without the budget of
/// the stages, the keys of the plan's ring and their coverage of the run's widths, and, when the keys report their
/// layout, the arena against the query. An input off the layout SlotsToCoeffs reads ([`ensure_chain_input`]) comes
/// first, then an S2C-first input without the budget SlotsToCoeffs consumes.
pub(crate) fn check_bootstrap_entry<BE, F, K>(
    module: &Module<BE>,
    res: &CKKSCiphertextOwned<BE>,
    a: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    available: usize,
) -> CKKSResult<()>
where
    BE: Backend,
    Module<BE>: CKKSBootstrappingOps<BE>,
    K: BootstrappingKeys<BE>,
{
    const OP: &str = "ckks_bootstrap";
    let run = ctx.run(a.meta());
    ensure_chain_input(OP, ctx.slots_to_coeffs().plan(), crate::GLWEPlaintextInfos::slot_layout(a))?;
    if ctx.pipeline() == BootstrappingPipeline::S2CFirst {
        checked_log_budget_sub(OP, a.log_budget(), ctx.slots_to_coeffs().consumed_bits())?;
    }
    ckks_ensure_operand!(
        OP,
        ctx.functional_message_modulus().is_none() && ctx.plan.eval_mod().eval_mod_type != EvalModType::ExpCmplx,
        "requires an identity bootstrapping context"
    );
    ensure_rank_one(OP, a, res)?;
    ensure_one_degree(OP, module.n(), a, res)?;
    ensure_bootstrap_degree(OP, ctx, a)?;
    ensure_one_base2k(OP, a, res)?;
    ensure_encapsulation(OP, keys, ctx.sparse_secret_hamming_weight())?;
    // A C2S-first raise lifts `a` itself.
    ctx.mod_raise_plan().scale_ups(OP, a.k().as_usize(), a.log_delta())?;
    let stages = ctx.stage_widths();
    let tensor_k = (
        stages.eval_mod_relinearization_k(res.k().as_usize(), a.log_delta()),
        stages.eval_mod_product_k(res.k().as_usize(), a.log_delta()),
    );
    match ctx.eval_mod_ci() {
        Some(_) => {
            let (tensor_key, trace_key, embed_key) = check_eval_mod_on_ci(OP, module, a, keys)?;
            check_key_widths(OP, res, a, ctx, keys, tensor_key, tensor_k, stages.eval_mod_bits())?;
            let widths = ctx.stage_widths().conversions(res.k().as_usize()).unwrap_or_default();
            check_conversion_key_widths(OP, trace_key, embed_key, widths)?;
        }
        None => {
            let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
            check_key_widths(OP, res, a, ctx, keys, tensor_key, tensor_k, stages.eval_mod_bits())?;
        }
    }
    ensure_eval_round_interval(&run)?;
    if let Some(keys_layout) = keys.keys_layout() {
        let bytes = module.ckks_bootstrap_tmp_bytes(res, a, ctx, keys_layout);
        ckks_ensure!(
            available >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {available}"
        );
    }
    Ok(())
}

/// What `ckks_functional_bootstrap_derived` refuses before its first step, as [`check_bootstrap_entry`], an input or a
/// SlotsToCoeffs plan off the full layout first, with the batch of LUTs, the message ratio, a plan with the EvalRound+ bypass, and the tensor key at the widths the LUTs
/// relinearize at.
pub(crate) fn check_functional_entry<BE, F, K>(
    module: &Module<BE>,
    res: &[CKKSCiphertextOwned<BE>],
    a: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    available: usize,
) -> CKKSResult<()>
where
    BE: Backend,
    Module<BE>: CKKSBootstrappingOps<BE>,
    K: BootstrappingKeys<BE>,
{
    const OP: &str = "ckks_functional_bootstrap";
    ensure_full_layout(OP, crate::GLWEPlaintextInfos::slot_layout(a))?;
    // SlotsToCoeffs, which runs first, reads its plan's layout: a replicated plan refuses the full-layout input.
    ensure_chain_input(OP, ctx.slots_to_coeffs().plan(), crate::GLWEPlaintextInfos::slot_layout(a))?;
    ckks_ensure_operand!(
        OP,
        ctx.pipeline() == BootstrappingPipeline::S2CFirst,
        "functional bootstrapping requires an S2C-first context"
    );
    ckks_ensure_operand!(
        OP,
        ctx.coeffs_to_slots_bypass().is_none(),
        "the functional bootstrap does not run the EvalRound+ bypass"
    );
    ensure_lut_batch(OP, module.n(), a, res, luts, &ctx.run(a.meta()))?;
    ensure_bootstrap_degree(OP, ctx, a)?;
    let log_msg_ratio = luts[0].log_msg_ratio();
    ensure_encapsulation(OP, keys, ctx.sparse_secret_hamming_weight())?;
    let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
    ensure_functional_message_ratio(OP, a, ctx.slots_to_coeffs().consumed_bits(), log_msg_ratio)?;
    ckks_ensure_operand!(
        OP,
        ctx.functional_message_modulus().unwrap_or(1usize << log_msg_ratio) == luts[0].message_modulus(),
        "functional bootstrap context must be configured for message modulus {}",
        luts[0].message_modulus()
    );
    for (ct_out, lut) in res.iter().zip(luts) {
        functional_output_contract(OP, a.meta(), ctx, lut, ct_out.k().as_usize())?;
    }
    // The LUT step runs on what CoeffsToSlots leaves of the raise, at the scale of the raised modulus, guard bits gone.
    let stages = ctx.stage_widths();
    let (log_delta, _) = stages.raise(a.k().as_usize());
    let k = res[0].k().as_usize().saturating_sub(ctx.coeffs_to_slots().consumed_bits());
    let log_budget = k.saturating_sub(log_delta + ctx.c2s_guard_bits());
    let tensor_k = luts
        .iter()
        .map(|lut| lut_relinearization_k(ctx.eval_mod_plan(), lut, log_budget, log_delta))
        .max()
        .unwrap_or(0);
    // The message ratio and the output contract above bound the LUT step on the raised scale.
    check_key_widths(OP, &res[0], a, ctx, keys, tensor_key, (tensor_k, tensor_k), 0)?;
    if let Some(keys_layout) = keys.keys_layout() {
        let bytes = module.ckks_functional_bootstrap_tmp_bytes(&res[0], a, ctx, luts, keys_layout);
        ckks_ensure!(
            available >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {available}"
        );
    }
    Ok(())
}

/// Widest ciphertext the LUT step relinearizes for `lut`, on an input of `log_budget` bits of budget at scale
/// `log_delta`, the guard bits gone: EvalMod's widest product when the LUT needs it, its own series' otherwise.
pub(crate) fn lut_relinearization_k<P>(
    eval_mod: &EvalModPlan,
    lut: &EncodedLut<P>,
    log_budget: usize,
    log_delta: usize,
) -> usize {
    if lut.requires_eval_mod() {
        eval_mod.relinearization_k(log_budget)
    } else {
        log_budget + log_delta
    }
}

/// The refusal of a bootstrap at `bootstrap_k` bits of an input of `input_k` bits that its stages do not fit.
fn narrower_than_stages(op: &'static str, bootstrap_k: usize, input_k: usize) -> CKKSError {
    let reason =
        format!("a bootstrap at {bootstrap_k} bits of an input of {input_k} bits is narrower than the bits its stages consume");
    CKKSCompositionError::InvalidOperand { op, reason }.into()
}

/// Fails before any work unless every key covers the widest ciphertext the bootstrap switches with it, the tensor key
/// `tensor_k` bits and decomposing at least `product_k`, and with `InvalidOperand` when the working width or the input
/// is narrower than the bits the stages running on it consume, the step after CoeffsToSlots `after_c2s_bits`.
///
/// The bootstrap runs at `ct_out.k()`, which need not be the width the keys
/// were laid out for. Each stage's key lookup also refuses a key too narrow for
/// its width; this check only moves the failure ahead of the computation. Keys
/// with a row offset (sparse-to-dense, first factor) are laid out for one raise:
/// they are checked against the raise this bootstrap runs. `op` names the
/// bootstrap in the errors.
#[expect(
    clippy::too_many_arguments,
    reason = "the check reads both ciphertexts, the keys, the tensor key's `k` and the bits after CoeffsToSlots"
)]
fn check_key_widths<BE, Rg, F, K, T, C1, C2>(
    op: &'static str,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    tensor_key: &T,
    (tensor_k, product_k): (usize, usize),
    after_c2s_bits: usize,
) -> CKKSResult<()>
where
    BE: Backend,
    Rg: Ring,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<BE, Rg>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (bootstrap_k, input_k) = (ct_out.k().as_usize(), ct_in.k().as_usize());
    let Some(starts) = ctx.stage_widths().stage_starts(bootstrap_k, input_k, after_c2s_bits) else {
        return Err(narrower_than_stages(op, bootstrap_k, input_k));
    };
    for key in ctx.rotation_key_offsets() {
        let k = TorusPrecision(starts.width(key) as u32);
        keys.get_automorphism_key(key.p, k)
            .map_err(|source| CKKSCompositionError::automorphism_key_lookup(op, key.p, k.into(), source))?;
    }
    let k = TorusPrecision(tensor_k as u32);
    let key = tensor_key
        .get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::tensor_key_lookup(op, k.into(), source))?;
    // EvalMod's product, wider on a replicated layout, decomposes through the key it looks up at `tensor_k`.
    let product = TorusPrecision(product_k as u32);
    ensure_key_covers(op, &key, product).map_err(|source| CKKSCompositionError::tensor_key_lookup(op, product.into(), source))?;
    check_raise_keys(op, ct_out, ct_in, ctx, keys)
}

/// The raise of `ct_in` into `ct_out.k()` bits, as its step checks it ([`check_raise_widths`]): fails before any work
/// unless the encapsulation and first-factor keys fit it.
fn check_raise_keys<BE, F, K, C1, C2>(
    op: &'static str,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
) -> CKKSResult<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (k_small, raised) = ctx.stage_widths().raise(ct_in.k().as_usize());
    let src = CKKSLayout {
        glwe_layout: GLWELayout {
            k: k_small.into(),
            ..ct_out.glwe_layout()
        },
        meta: GLWEPlaintextMeta::default(),
    };
    check_raise_widths(op, ct_out, &src, raised - k_small, ctx.coeffs_to_slots_first_factor(), keys)
}

/// What the raise refuses on widths, labelled `op`: `src`, lifted by `scale_up` fused
/// into ModUp, into `dst`, under the encapsulation keys and the first-factor keys of `keys`.
pub(crate) fn check_raise_widths<BE, Dst, Src, K>(
    op: &'static str,
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
    keys: &K,
) -> CKKSResult<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
{
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if let Some(factor) = first_factor {
        if let Some((dense_to_sparse, _)) = keys.encapsulation_keys() {
            ensure_encapsulation_key_covers(op, src.k(), dense_to_sparse)?;
        }
        ensure_mod_up_widens(op, k_large, k_small, scale_up)?;
        check_first_factor_raise(
            op,
            dst.base2k().as_usize(),
            dst.n().as_usize().ilog2() as usize,
            k_large,
            k_small + scale_up,
            factor,
            keys,
        )?;
        return Ok(());
    }
    match keys.encapsulation_keys() {
        Some((dense_to_sparse, Some(sparse_to_dense))) => {
            check_encapsulated_raise(op, dst, src, scale_up, dense_to_sparse, sparse_to_dense)
        }
        Some((_, None)) => Err(CKKSCompositionError::InvalidOperand {
            op,
            reason:
                "the encapsulated ModUp needs a sparse-to-dense key, which a key set for the first factor on the raise has not"
                    .into(),
        }
        .into()),
        None => ensure_mod_up_widens(op, k_large, k_small, scale_up),
    }
}

/// `InvalidOperand` unless the encapsulation key `key` takes an input of `k` bits.
fn ensure_encapsulation_key_covers<K: GGLWEInfos>(op: &'static str, k: TorusPrecision, key: &K) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        k <= key.gglwe_layout().max_input_k(),
        "the input is wider than an encapsulation key's max_input_k"
    );
    Ok(())
}

/// Refuses encapsulation keys too narrow for their inputs before any write: `src`
/// for the dense-to-sparse switch, `dst` for an offset-free sparse-to-dense one, and
/// the raise of `src` by `scale_up` into `dst`, a sparse-to-dense key with a row
/// offset being checked against that raise, by [`check_offset_raise`].
pub(crate) fn check_encapsulated_raise<Dst: CKKSCtBounds, Src: CKKSCtBounds, D2S: GGLWEInfos, S2D: GGLWEInfos>(
    op: &'static str,
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
) -> CKKSResult<()> {
    ensure_encapsulation_key_covers(op, src.k(), dense_to_sparse)?;
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if sparse_to_dense.k_offset().as_usize() > 0 {
        check_offset_raise(op, k_large, k_small, scale_up, sparse_to_dense)?;
    } else {
        ensure_encapsulation_key_covers(op, dst.k(), sparse_to_dense)?;
        ensure_mod_up_widens(op, k_large, k_small, scale_up)?;
    }
    Ok(())
}

/// Bits of input a sparse-to-dense key with a row offset takes for a raise
/// into `dst_k` bits: those below its offset, which its rows must reach.
pub(crate) fn offset_key_window<K: GGLWEInfos>(dst_k: usize, key: &K) -> usize {
    dst_k.saturating_sub(key.k_offset().as_usize())
}

/// `InvalidOperand` for `op` unless the raise of a `k_small`-bit ciphertext lifted by `scale_up` into `k_large` bits
/// widens it.
pub(crate) fn ensure_mod_up_widens(op: &'static str, k_large: usize, k_small: usize, scale_up: usize) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        k_large >= k_small + scale_up,
        "dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up}); ModUp must widen, not shrink, the modulus"
    );
    Ok(())
}

/// Checks that a sparse-to-dense key with a row offset serves the raise of a
/// `k_small`-bit ciphertext lifted by `scale_up` into `k_large` bits, and
/// returns the raise's shift and the bits the key's window then holds.
pub(crate) fn check_offset_raise<K: GGLWEInfos>(
    op: &'static str,
    k_large: usize,
    k_small: usize,
    scale_up: usize,
    key: &K,
) -> CKKSResult<(usize, usize)> {
    ensure_mod_up_widens(op, k_large, k_small, scale_up)?;
    let shift = k_large - k_small - scale_up;
    let k_offset = key.k_offset().as_usize();
    ckks_ensure_operand!(
        op,
        k_offset <= shift,
        "the sparse-to-dense key lowers its input by {k_offset} bits, more than the raise's {shift}"
    );
    let window = offset_key_window(k_large, key) - scale_up;
    let max_input_k = key.gglwe_layout().max_input_k().as_usize();
    let reach = k_offset + key.gglwe_layout().gadget_k().as_usize();
    ckks_ensure_operand!(
        op,
        window <= max_input_k && k_large <= reach,
        "the sparse-to-dense key takes {max_input_k} bits of input and reaches bit {reach}; the raise needs {window} bits up to bit {k_large}"
    );
    Ok((shift, window))
}

/// `key`, the tensor key of the plan's ring, or the refusal of `op`.
pub(crate) fn required_tensor_key<'a, T>(op: &'static str, key: Option<&'a T>) -> CKKSResult<&'a T> {
    key.ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: "the keys hold no tensor key of the plan's ring".into(),
        }
        .into()
    })
}

/// `log_delta` without the plan's `guard_bits`, or `InvalidOperand` for `op` when it holds fewer.
pub(crate) fn remove_guard_bits(op: &'static str, log_delta: usize, guard_bits: usize) -> CKKSResult<usize> {
    log_delta.checked_sub(guard_bits).ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: "the input holds fewer scale bits than the plan's guard bits".into(),
        }
        .into()
    })
}

/// The plan constructor rejects this, but re-check before the power-of-two shift of EvalRound+.
pub(crate) fn ensure_eval_round_interval<BE, F, P>(plan: &P) -> CKKSResult<()>
where
    BE: Backend,
    P: CKKSEvalModInfos<BE, F>,
{
    if plan.eval_round_plus() {
        ckks_ensure!(
            plan.eval_mod_plan().f_mod_interval.is_power_of_two(),
            "EvalRound+ requires a power-of-two f_mod_interval, got {}",
            plan.eval_mod_plan().f_mod_interval
        );
    }
    Ok(())
}

/// EvalMod on the standard ring, which the two standard routes run.
pub(crate) fn eval_mod_standard<'a, BE, F, P>(op: &'static str, plan: &'a P) -> CKKSResult<&'a EvalMod<F, CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    P: CKKSEvalModInfos<BE, F>,
{
    plan.eval_mod_standard().ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: "the plan runs no EvalMod on the standard ring".into(),
        }
        .into()
    })
}

/// The conversion keys, or the refusal of `op`.
fn required_conversion_keys<'a, BE, K>(op: &'static str, keys: &'a K) -> CKKSResult<(&'a K::TraceKey, &'a K::EmbedKey)>
where
    BE: Backend,
    K: BootstrappingKeys<BE>,
{
    keys.conversion_keys().ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: "the keys hold no conversion keys of the plan's ring".into(),
        }
        .into()
    })
}

/// Refuses what the query and the body of EvalMod on the conjugate-invariant ring take on trust: a module of another
/// degree than `2N`, the degree its compilation reads, and absent keys or conversion keys of another shape. Returns
/// the tensor key of that ring and the conversion keys.
pub(crate) fn check_eval_mod_on_ci<'a, BE, A, K>(
    op: &'static str,
    module: &Module<BE>,
    a: &A,
    keys: &'a K,
) -> CKKSResult<(&'a K::CITensorKey, &'a K::TraceKey, &'a K::EmbedKey)>
where
    BE: Backend,
    A: CKKSCtBounds,
    K: BootstrappingKeys<BE>,
{
    ckks_ensure_operand!(op, module.n() == 2 * a.n().as_usize(), "takes a module of degree 2N");
    let tensor_key = required_tensor_key(op, keys.ci_tensor_key())?;
    let (trace_key, embed_key) = check_conversion_key_shapes(op, a, keys)?;
    Ok((tensor_key, trace_key, embed_key))
}

/// Refuses conversion keys of another degree, radix or rank than the ciphertexts they switch,
/// or absent: they go through no key source, which would check their shapes and coverage.
fn check_conversion_key_shapes<'a, BE, K, A>(
    op: &'static str,
    a: &A,
    keys: &'a K,
) -> CKKSResult<(&'a K::TraceKey, &'a K::EmbedKey)>
where
    BE: Backend,
    K: BootstrappingKeys<BE>,
    A: CKKSCtBounds,
{
    let (trace_key, embed_key) = required_conversion_keys(op, keys)?;
    let n_wide = 2 * a.n().as_usize();
    for (name, key, view) in [
        ("trace key", trace_key.gglwe_layout(), trace_key.to_backend_ref()),
        ("embed key", embed_key.gglwe_layout(), embed_key.to_backend_ref()),
    ] {
        validate_gadget_backend_view(name, &key, &view, n_wide, a.base2k(), 0).map_err(|error| {
            CKKSCompositionError::InvalidOperand {
                op,
                reason: error.to_string(),
            }
        })?;
    }
    Ok((trace_key, embed_key))
}

/// Refuses conversion keys narrower than the ciphertexts they switch, of `traced` bits to the
/// conjugate-invariant ring and of `embedded` bits back.
pub(crate) fn check_conversion_key_widths<T, E>(
    op: &'static str,
    trace_key: &T,
    embed_key: &E,
    (traced, embedded): (usize, usize),
) -> CKKSResult<()>
where
    T: GGLWEInfos,
    E: GGLWEInfos,
{
    for (key, k) in [(trace_key.gglwe_layout(), traced), (embed_key.gglwe_layout(), embedded)] {
        let covered = key.max_input_k().as_usize();
        if k > covered {
            let reason = format!("a conversion key covers inputs of {covered} bits, asked for {k}");
            return Err(CKKSCompositionError::InvalidOperand { op, reason }.into());
        }
    }
    Ok(())
}

fn ensure_unit_circle_exp_context(op: &'static str, plan: &EvalModPlan) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        plan.eval_mod_type == EvalModType::ExpCmplx,
        "general LUT evaluation requires an ExpCmplx EvalMod plan"
    );
    ckks_ensure_operand!(
        op,
        plan.scaling == Some(std::f64::consts::TAU),
        "general LUT evaluation requires a unit-circle ExpCmplx plan (scaling = 2π)"
    );
    Ok(())
}

/// What a batch of LUTs asks of its operands, before any write: at least one LUT and as many outputs, one message
/// modulus, a unit-circle complex exponential on the standard ring when a LUT needs EvalMod, and one rank-1
/// layout for the input and the outputs.
pub(crate) fn ensure_lut_batch<BE, F, P, A, R>(
    op: &'static str,
    module_n: usize,
    a: &A,
    outs: &[R],
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    plan: &P,
) -> CKKSResult<()>
where
    BE: Backend,
    P: CKKSEvalModInfos<BE, F>,
    A: CKKSCtBounds,
    R: CKKSCtBounds,
{
    ckks_ensure_operand!(op, !luts.is_empty(), "needs at least one LUT");
    ckks_ensure_operand!(
        op,
        outs.len() == luts.len(),
        "the number of outputs differs from the number of LUTs ({} vs {})",
        outs.len(),
        luts.len()
    );
    ckks_ensure_operand!(
        op,
        luts.iter().all(|lut| lut.message_modulus() == luts[0].message_modulus()),
        "LUTs must have the same message modulus"
    );
    ckks_ensure_operand!(
        op,
        luts.iter()
            .all(|lut| crate::GLWEPlaintextInfos::slot_layout(lut.coeffs()) == crate::GLWEPlaintextInfos::slot_layout(a)),
        "the constants are encoded for another slot layout than the input's"
    );
    if luts.iter().any(EncodedLut::requires_eval_mod) {
        ensure_unit_circle_exp_context(op, plan.eval_mod_plan())?;
        eval_mod_standard(op, plan)?;
    }
    let head = &outs[0];
    ckks_ensure_operand!(
        op,
        a.rank().as_usize() == 1
            && outs.iter().all(|ct| {
                ct.rank().as_usize() == 1 && ct.n() == head.n() && ct.base2k() == head.base2k() && ct.k() == head.k()
            }),
        "LUT outputs must share one rank-1 layout"
    );
    ensure_one_base2k(op, a, head)?;
    ensure_one_degree(op, module_n, a, head)
}

/// What the output of `lut` carries from a raise at `bootstrap_k` bits of `message`: its metadata and width.
pub(crate) fn functional_output_contract<BE: Backend, F, Rg: Ring>(
    op: &'static str,
    message: GLWEPlaintextMeta,
    ctx: &BootstrappingContextPrepared<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE, Rg>>,
    bootstrap_k: usize,
) -> CKKSResult<(GLWEPlaintextMeta, usize)> {
    let log_delta = ckks_log_delta(&message)
        .checked_add(lut.log_msg_ratio())
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap output scale overflow")))?;
    let eval_mod = if lut.requires_eval_mod() {
        ctx.eval_mod_plan().consumed_bits()
    } else {
        0
    };
    let consumed = ctx
        .coeffs_to_slots()
        .consumed_bits()
        .checked_add(ctx.c2s_guard_bits())
        .and_then(|bits| bits.checked_add(eval_mod))
        .and_then(|bits| bits.checked_add(lut.consumed_bits(log_delta)))
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap budget overflow")))?;
    ckks_ensure_operand!(
        op,
        bootstrap_k >= consumed,
        "functional bootstrap output width {bootstrap_k} is smaller than its {consumed}-bit post-ModUp cost"
    );
    let output_k = bootstrap_k - consumed;
    ckks_ensure_operand!(
        op,
        output_k >= log_delta,
        "functional bootstrap leaves output k={output_k} below log_delta={log_delta}"
    );
    Ok((
        GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            ..message
        },
        output_k,
    ))
}

fn ensure_functional_message_ratio<C>(op: &'static str, ct_in: &C, s2c_consumed_bits: usize, expected: usize) -> CKKSResult<()>
where
    C: CKKSInfos,
{
    let actual = checked_log_budget_sub(op, ct_in.log_budget(), s2c_consumed_bits)?;
    ckks_ensure_operand!(
        op,
        actual == expected,
        "functional bootstrap LUT requires log_msg_ratio {expected}, got {actual}"
    );
    Ok(())
}

/// Bits a diagonal's product adds once divided by its scale: its encoded width
/// past its scale.
fn diagonal_growth<P: IntPolyInfos + CKKSInfos>(diagonal: &P) -> usize {
    diagonal.encoded_k().as_usize() - diagonal.log_delta()
}

/// Limbs of the small ciphertext a first-factor key switches: its rows sit
/// under it and end at the working width.
pub(crate) fn small_limbs<K: GGLWEInfos>(op: &'static str, base2k: usize, k_large: usize, key: &K) -> CKKSResult<usize> {
    let (work, _) = first_factor_work(base2k, k_large);
    let k_offset = key.k_offset().as_usize();
    ckks_ensure_operand!(
        op,
        key.base2k().as_usize() == base2k && k_offset.is_multiple_of(base2k) && k_offset < work * base2k,
        "a first-factor key lowers its input by {k_offset} bits, not a whole number of {base2k}-bit limbs below the raise's {work} limbs"
    );
    let small = work - k_offset / base2k;
    let layout = key.gglwe_layout();
    let reach = k_offset + layout.gadget_k().as_usize();
    ckks_ensure_operand!(
        op,
        layout.max_input_k().as_usize() >= small * base2k && reach >= k_large,
        "a first-factor key takes {} bits of input and reaches bit {reach}; the raise switches {} bits up to bit {k_large}",
        layout.max_input_k(),
        small * base2k
    );
    Ok(small)
}

/// The first-factor key of the rotation `rotation`, `InvalidOperand` for `op` when `keys` has none.
pub(crate) fn first_factor_key<'k, BE: Backend, K: BootstrappingLinearKeys<BE>>(
    op: &'static str,
    keys: &'k K,
    galois: i64,
) -> CKKSResult<&'k K::SwitchingKey> {
    let Some(key) = keys.first_factor_key(galois) else {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: format!("no first-factor key for the Galois element {galois}"),
        }
        .into());
    };
    Ok(key)
}

/// What [`check_first_factor_raise`] finds a raise with the first factor to
/// work with.
pub(crate) struct FirstFactorRaise {
    /// Layout of every first-factor key.
    pub(crate) key_layout: GGLWELayout,
    /// Limbs of the terms the keys switch.
    pub(crate) small: usize,
    /// Bits a product by any diagonal adds once divided by its scale.
    pub(crate) growth: usize,
}

/// Checks that `factor` and the first-factor keys of `keys` serve a raise into
/// `k_large` bits at radix `base2k` of a ciphertext that fills `raised` bits
/// once lifted: one diagonal per giant step and no baby step, all diagonals of
/// one scale and width, and one key per rotation, all of one layout that fits
/// the raise. The factor's shape is the compiled context's invariant; the keys' presence, layouts and coverage are
/// `InvalidOperand` for `op`.
pub(crate) fn check_first_factor_raise<BE, K>(
    op: &'static str,
    base2k: usize,
    log_n: usize,
    k_large: usize,
    raised: usize,
    factor: &LinearTransformation<CKKSRightOperandPrepared<BE>>,
    keys: &K,
) -> CKKSResult<FirstFactorRaise>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
{
    let Some(diagonal) = factor.first_diagonal_plaintext() else {
        ckks_bail!("{op}: the first factor has no diagonal");
    };
    let (growth, width) = (diagonal_growth(diagonal), diagonal.encoded_k());
    let key_layout = first_factor_key(op, keys, factor.giant_steps[0].galois)?.gglwe_layout();
    for giant in &factor.giant_steps {
        ckks_ensure!(
            giant.diagonals.len() == 1 && galois_is_identity(giant.diagonals[0].baby_galois),
            "{op}: a first factor on the raise has one diagonal per rotation and no baby step"
        );
        let diagonal = &giant.diagonals[0].plaintext;
        ckks_ensure!(
            diagonal_growth(diagonal) == growth && diagonal.encoded_k() == width,
            "{op}: the diagonals of the first factor differ in scale or width"
        );
        ckks_ensure_operand!(
            op,
            first_factor_key(op, keys, giant.galois)?.gglwe_layout() == key_layout,
            "the first-factor key of the Galois element {} differs in layout from the others",
            giant.galois
        );
    }
    let small = small_limbs(op, base2k, k_large, &key_layout)?;
    let needed = first_factor_small_bits(base2k, log_n, k_large, raised, growth);
    ckks_ensure_operand!(
        op,
        needed <= small * base2k,
        "the first-factor keys switch {} bits, the raise of a ciphertext filling {raised} bits needs {needed}",
        small * base2k
    );
    Ok(FirstFactorRaise {
        key_layout,
        small,
        growth,
    })
}

/// The refusals that the default bodies of `ckks_add_one_assign` and `ckks_sub_one_assign` make, under `op`, the
/// shifting method, for `dst` on the ring `Rg`, with `bytes` their query (`ckks_add_one_tmp_bytes` or
/// `ckks_sub_one_tmp_bytes` of `dst.max_size()`). An override that keeps its unit plaintext runs them before using it.
///
/// # Errors
///
/// The refusals of building the unit plaintext of the slot layout of `dst`, then those of the plaintext route of
/// `ckks_add_assign` adding it to `dst`, each carrying `op`, then `Internal` for a `scratch` below `bytes`.
pub fn check_unit_shift<BE: Backend, Dst: CKKSCtBounds, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    dst: &Dst,
    scratch: &ScratchArena<'_, BE>,
    bytes: usize,
) -> CKKSResult<()> {
    checked_one::<BE, Dst, Rg>(op, module, dst)?;
    ensure_scratch(op, scratch, bytes)
}

/// The unit coefficients of the slot layout of `dst`, after the refusals of their plaintext addition to `dst` under `op`.
pub(crate) fn checked_one<BE: Backend, Dst: CKKSCtBounds, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    dst: &Dst,
) -> CKKSResult<OneCoeffs> {
    let one = one_coeffs::<Rg>(op, dst.slot_layout(), dst.log_delta())?;
    let layout = one.layout(dst.base2k());
    check_add_pt(op, module, dst, &layout)?;
    ensure_plaintext_operand(op, dst.base2k().as_usize(), dst.n().as_usize(), dst.log_budget(), &layout)?;
    Ok(one)
}
