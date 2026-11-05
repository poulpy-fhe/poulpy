//! Validation shared across modules: generic key and operand layouts, and the refusals of the bootstrap steps and of the
//! bodies composing them.
//!
//! The generic helpers check the structural invariants (degree, rank, radix, decomposition shape, storage capacity, backend-view consistency) that any gadget key or runtime ciphertext operand must satisfy before entering the arithmetic primitives, whichever pipeline consumes them, and the uniformity of a linear transformation's diagonals.
//! The bootstrap refusals sit here because `oep::derived` and `reference` both read them, and `oep` names no `reference`; a check only one `reference` module reads, as PaCo's plan-against-material checks, stays there.

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    LinearTransformationPrepared,
    layouts::{
        Base2K, GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LWEInfos,
        LinearTransformation, TorusPrecision, prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSError, CKKSInfos, CKKSResult, GLWEPlaintextMeta, Scale, SlotsKind,
    api::CKKSBootstrappingOps,
    checked_log_budget_sub, ckks_bail, ckks_ensure, ckks_ensure_operand, ckks_log_delta,
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingLinearKeys, BootstrappingPipeline, CKKSCITwin,
        CKKSCiphertextOwned, CKKSEvalModInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared, EncodedLut, EvalMod, EvalModPlan,
        EvalModType,
    },
    presets::first_factor::{first_factor_small_bits, first_factor_work},
};

/// `InvalidOperand` for `op` unless `prepared` was allocated for `lt`'s layout: the same schedule, every diagonal of
/// both at one degree, radix, rank, width and prepared limb count (a ciphertext diagonal's live limbs, a plaintext's
/// allocated ones), and `lt`'s baby steps led by the identity rotation.
pub(crate) fn ensure_prepared_transformation_layout<BE: Backend, P: GLWEInfos>(
    op: &'static str,
    prepared: &LinearTransformationPrepared<BE>,
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
        (p.glwe_layout(), if p.rank().as_usize() > 0 { p.size() } else { p.max_size() })
    });
    if prepared_layouts.chain(source_layouts).any(|diagonal| diagonal != layout) {
        return Err(invalid("a diagonal's layout differs from the prepared transformation's"));
    }
    if lt.baby_steps.first().is_some_and(|&baby| baby != 0) {
        return Err(invalid("the baby steps do not start with the identity rotation"));
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
                        "linear transformation diagonals are not uniform: diagonal (giant rot {}, baby {}) has (log_scale {diag_log_scale}, max_k {diag_max_k}, {diag_slots:?}) but the first diagonal has ({log_scale}, {max_k}, {slots:?})",
                        gs.rot, diag.baby,
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
pub(crate) fn validate_backend_storage_capacity<BE, K>(name: &str, key: &K) -> Result<()>
where
    BE: Backend,
    K: GLWEInfos + GLWEToBackendRef<BE>,
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

/// `InvalidOperand` for `op` unless a CI module comes with a plan of EvalMod on the CI ring, and only with one.
pub(crate) fn ensure_ci_module<M>(op: &'static str, ci: Option<&M>, plan_on_ci: bool) -> CKKSResult<()> {
    ckks_ensure_operand!(
        op,
        ci.is_some() == plan_on_ci,
        "a CI module goes with a plan of EvalMod on the CI ring, and only with one"
    );
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

/// What `ckks_bootstrap_derived` refuses before its first step, so that no later step refuses with `res` half written:
/// the context kind, the module of the plan's ring, ranks, degrees and radices, a working width without the budget of
/// the stages running on `res`, the keys of the plan's ring and their coverage of the run's widths, and, when the keys
/// report their layout, the arena against the query.
pub(crate) fn check_bootstrap_entry<BE, F, K>(
    module: &Module<BE>,
    res: &CKKSCiphertextOwned<BE>,
    a: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys: &K,
    available: usize,
) -> CKKSResult<()>
where
    BE: CKKSCITwin,
    Module<BE>: CKKSBootstrappingOps<BE>,
    K: BootstrappingKeys<BE>,
{
    const OP: &str = "ckks_bootstrap";
    ckks_ensure_operand!(
        OP,
        ctx.functional_message_modulus().is_none(),
        "requires an identity bootstrapping context"
    );
    ensure_ci_module(OP, ci, ctx.eval_mod_ci().is_some())?;
    ensure_rank_one(OP, a, res)?;
    ensure_one_degree(OP, module.n(), a, res)?;
    ensure_one_base2k(OP, a, res)?;
    ensure_encapsulation(OP, keys, ctx.sparse_secret_hamming_weight())?;
    // `key_widths` reads widths: SlotsToCoeffs on `res` also needs its budget above the scale EvalMod leaves there.
    if ctx.pipeline() == BootstrappingPipeline::C2SFirst
        && res.k().as_usize() < ctx.plan.post_mod_up_consumed_bits() + ctx.plan.eval_mod().f_mod_log_delta
    {
        return Err(narrower_than_stages(OP, res.k().as_usize(), a.k().as_usize()));
    }
    match ci {
        Some(ci) => {
            let (tensor_key, trace_key, embed_key) = check_eval_mod_on_ci(OP, module, ci, a, keys)?;
            check_key_widths(OP, res, a, ctx, keys, tensor_key)?;
            let widths = ctx.stage_widths().conversions(res.k().as_usize()).unwrap_or_default();
            check_conversion_key_widths(OP, trace_key, embed_key, widths)?;
        }
        None => check_key_widths(OP, res, a, ctx, keys, required_tensor_key(OP, keys.tensor_key())?)?,
    }
    ensure_eval_round_interval(&ctx.run(a.meta()))?;
    if let Some(keys_layout) = keys.keys_layout() {
        let bytes = module.ckks_bootstrap_tmp_bytes(res, a, ctx, ci, keys_layout);
        ckks_ensure!(
            available >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {available}"
        );
    }
    Ok(())
}

/// What `ckks_functional_bootstrap_derived` refuses before its first step, as [`check_bootstrap_entry`], with the
/// batch of LUTs and the message ratio.
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
    BE: CKKSCITwin,
    Module<BE>: CKKSBootstrappingOps<BE>,
    K: BootstrappingKeys<BE>,
{
    const OP: &str = "ckks_functional_bootstrap";
    ckks_ensure_operand!(
        OP,
        ctx.pipeline() == BootstrappingPipeline::S2CFirst,
        "functional bootstrapping requires an S2C-first context"
    );
    ensure_lut_batch(OP, module.n(), a, res, luts, &ctx.run(a.meta()))?;
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
    check_key_widths(OP, &res[0], a, ctx, keys, tensor_key)?;
    if let Some(keys_layout) = keys.keys_layout() {
        let bytes = module.ckks_functional_bootstrap_tmp_bytes(&res[0], a, ctx, luts, keys_layout);
        ckks_ensure!(
            available >= bytes,
            "{OP} needs a scratch arena of {bytes} bytes, got {available}"
        );
    }
    Ok(())
}

/// The refusal of a bootstrap at `bootstrap_k` bits of an input of `input_k` bits that its stages do not fit.
fn narrower_than_stages(op: &'static str, bootstrap_k: usize, input_k: usize) -> CKKSError {
    let reason =
        format!("a bootstrap at {bootstrap_k} bits of an input of {input_k} bits is narrower than the bits its stages consume");
    CKKSCompositionError::InvalidOperand { op, reason }.into()
}

/// Fails before any work unless every key covers the widest ciphertext the
/// bootstrap switches with it, and with `InvalidOperand` when the working width
/// or the input is narrower than the bits the stages running on it consume.
///
/// The bootstrap runs at `ct_out.k()`, which need not be the width the keys
/// were laid out for. Each stage's key lookup also refuses a key too narrow for
/// its width; this check only moves the failure ahead of the computation. Keys
/// with a row offset (sparse-to-dense, first factor) are laid out for one raise:
/// they are checked against the raise this bootstrap runs. `op` names the
/// bootstrap in the errors.
fn check_key_widths<BE, M, F, K, T, C1, C2>(
    op: &'static str,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    tensor_key: &T,
) -> CKKSResult<()>
where
    BE: Backend,
    M: Backend,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<M>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let log_n = ct_out.n().as_usize().trailing_zeros() as usize;
    let (bootstrap_k, input_k) = (ct_out.k().as_usize(), ct_in.k().as_usize());
    let stages = ctx.stage_widths();
    let Some(starts) = stages.stage_starts(bootstrap_k, input_k) else {
        return Err(narrower_than_stages(op, bootstrap_k, input_k));
    };
    let other_degree;
    let uses = match ctx.rotation_key_uses(log_n) {
        Some(uses) => uses,
        None => {
            other_degree = stages.rotation_key_uses(log_n);
            &other_degree
        }
    };
    for key_uses in uses.chunk_by(|a, b| a.p == b.p) {
        let p = key_uses[0].p;
        let k = key_uses.iter().map(|key_use| starts.width(key_use)).max().unwrap_or(0);
        let k = TorusPrecision(k as u32);
        keys.rotation_keys()
            .get_automorphism_key(p, k)
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation: p,
                k: k.into(),
                source,
            })?;
    }
    let k = TorusPrecision(stages.eval_mod_relinearization_k(bootstrap_k, ct_in.log_delta()) as u32);
    tensor_key
        .get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source })?;
    check_raise_keys(op, ct_out, ct_in, ctx, keys)
}

/// Keys with a row offset are laid out for the raise: fails before any work
/// unless the first-factor keys, or a sparse-to-dense key with an offset, fit
/// the raise of `ct_in` into `ct_out.k()` bits.
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
    let bootstrap_k = ct_out.k().as_usize();
    let (k_small, raised) = ctx.stage_widths().raise(ct_in.k().as_usize());
    if let Some(factor) = ctx.coeffs_to_slots_first_factor() {
        check_first_factor_raise(
            op,
            ct_out.base2k().as_usize(),
            ct_out.n().as_usize().ilog2() as usize,
            bootstrap_k,
            raised,
            factor,
            keys,
        )?;
    } else if let Some((_, Some(sparse_to_dense))) = keys.encapsulation_keys()
        && sparse_to_dense.k_offset().as_usize() > 0
    {
        check_offset_raise(op, bootstrap_k, k_small, raised - k_small, sparse_to_dense)?;
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
    BE: CKKSCITwin,
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
    BE: CKKSCITwin,
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
    BE: CKKSCITwin,
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

/// Refuses what the query and the body of EvalMod on the conjugate-invariant ring take on trust: modules of other
/// degrees than `2N` and `N`, and absent keys or conversion keys of another shape. Returns the tensor key of that ring
/// and the conversion keys.
pub(crate) fn check_eval_mod_on_ci<'a, BE, A, K>(
    op: &'static str,
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    a: &A,
    keys: &'a K,
) -> CKKSResult<(&'a K::CITensorKey, &'a K::TraceKey, &'a K::EmbedKey)>
where
    BE: CKKSCITwin,
    A: CKKSCtBounds,
    K: BootstrappingKeys<BE>,
{
    ckks_ensure_operand!(
        op,
        module.n() >= 2 * a.n().as_usize() && ci.n() == a.n().as_usize(),
        "takes a module of degree at least 2N and a CI module of degree N"
    );
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
    BE: CKKSCITwin,
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
    BE: CKKSCITwin,
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
pub(crate) fn functional_output_contract<BE: Backend, F>(
    op: &'static str,
    message: GLWEPlaintextMeta,
    ctx: &BootstrappingContextPrepared<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
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
    rotation: i64,
) -> CKKSResult<&'k K::SwitchingKey> {
    let Some(key) = keys.first_factor_key(rotation) else {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: format!("no first-factor key for the rotation {rotation}"),
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
    let key_layout = first_factor_key(op, keys, factor.giant_steps[0].rot)?.gglwe_layout();
    for giant in &factor.giant_steps {
        ckks_ensure!(
            giant.diagonals.len() == 1 && giant.diagonals[0].baby == 0,
            "{op}: a first factor on the raise has one diagonal per rotation and no baby step"
        );
        let diagonal = &giant.diagonals[0].plaintext;
        ckks_ensure!(
            diagonal_growth(diagonal) == growth && diagonal.encoded_k() == width,
            "{op}: the diagonals of the first factor differ in scale or width"
        );
        ckks_ensure_operand!(
            op,
            first_factor_key(op, keys, giant.rot)?.gglwe_layout() == key_layout,
            "the first-factor key of the rotation {} differs in layout from the others",
            giant.rot
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
