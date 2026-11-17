//! The steps of the replicated route of a bootstrap ([`ReplicatedRoute`](crate::layouts::ReplicatedRoute)): the
//! traces, working bits and roundings around the factor chains of its transforms, and EvalMod on the two parts of
//! CoeffsToSlots.

use poulpy_core::{
    GLWETrace,
    layouts::{GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, SetK},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSImagOps, CKKSRotateOps, CKKSRoundOps,
        CKKSSubOps,
    },
    layouts::{
        CKKSCiphertextViewMut, CKKSDFTFactors, CKKSPlaintextOwned, Decode, Encode, EvalMod, MaskedCoeffs, ReplicatedStage,
        ScratchArenaTakeCKKSCiphertext, Standard, bootstrapping::replicated::ReplicatedStep,
    },
};

/// `ct` at its own width and scale, through the normalized trace from level `skip`, computed `extra_bits` under its
/// scale, and rounded back with the shaping of its layout. The average of the copies keeps the scale and the budget.
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
    let working = GLWELayout {
        k: (ct.k().as_usize() + extra_bits).into(),
        ..ct.glwe_layout()
    };
    let (mut wide, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, ct.meta());
    module.ckks_copy(&mut wide, &*ct, &mut scratch)?;
    wide.set_log_delta(ct.log_delta() + extra_bits);
    let log_n = wide.n().log2();
    module.glwe_trace_assign(&mut wide, skip..log_n, false, true, keys, &mut scratch);
    module.ckks_round(ct, &wide, &mut scratch)
}

/// SlotsToCoeffs of a replicated run, in place on `ct`, a copy of the bootstrap's input at `Δ_b` or at the leveled
/// scale, allocated at least as wide as the input width of the route.
///
/// The input is brought to `Δ_b` at the input width, its copies of the period averaged by the normalized trace onto
/// period `kR`, and the gather `dft` applied `extra_bits` under the scale, then rounded plainly to the message ratio.
/// `ct` ends at `Δ_b` with that ratio as its budget, holding the coefficients of the subring of degree `2k`: the full
/// layout, complex, with no sparsity.
pub(crate) fn slots_to_coeffs<BE, D, C, H>(
    module: &Module<BE>,
    ct: &mut C,
    dft: &D,
    stage: ReplicatedStage<'_>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSDFTOps<BE> + CKKSCopyOps<BE> + CKKSRoundOps<BE> + GLWETrace<BE>,
    D: CKKSDFTFactors<BE, Decode, Standard>,
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_slots_to_coeffs_assign";
    let route = stage.route;
    let k_in = route.log_delta + stage.log_msg_ratio + route.s2c_consumed_bits();
    crate::ckks_ensure_operand!(
        OP,
        ct.slot_layout() == SlotLayout::Replicated(route.layout),
        "a replicated run takes the layout of its route"
    );
    let log_delta = ct.log_delta();
    crate::ckks_ensure_operand!(
        OP,
        log_delta == route.log_delta || (log_delta == route.level_log_delta && log_delta < route.log_delta),
        "a replicated run takes an input at the bootstrapping or the leveled scale, got 2^{log_delta}"
    );
    crate::ckks_ensure_operand!(
        OP,
        ct.log_budget() + route.log_delta >= k_in,
        "the input has less budget than the input width of the route"
    );
    // The trace of core looks its keys up and switches with them without a refusal of its own.
    let lookups = module
        .glwe_trace_galois_elements(ct.n(), stage.trace_skip..ct.n().log2(), false)
        .into_iter()
        .map(|p| (p, k_in + route.extra_bits));
    crate::layouts::validation::ensure_galois_keys(OP, keys, lookups)?;
    let chain = crate::layouts::validation::chain_galois_lookups(
        OP,
        module,
        ct.n().as_usize(),
        dft.factor_plans(),
        stage.consumed,
        k_in + route.extra_bits,
    )?;
    crate::layouts::validation::ensure_galois_keys(OP, keys, chain)?;
    let lift = route.log_delta - log_delta;
    let at = |k: usize| GLWELayout {
        k: k.into(),
        ..ct.glwe_layout()
    };
    // The copies this step and its wide trace carve, at most `extra_bits` above the input width.
    let ct_bytes = |k: usize| crate::oep::derived::replicated::ct_bytes::<BE>(&at(k));
    let carved = ct_bytes(k_in) + 2 * ct_bytes(k_in + route.extra_bits);
    crate::ckks_ensure!(
        scratch.available() >= carved,
        "{OP} needs a scratch arena of at least {carved} bytes, got {}",
        scratch.available()
    );
    let (mut input, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&at(k_in), ct.meta());
    let (mut x, mut scratch) = scratch.take_ckks_ciphertext_scratch(&at(k_in + route.extra_bits), ct.meta());

    // A leveled input relabelled at `Δ_b`, the same torus value, at the input width.
    input.set_k((k_in - lift).into());
    module.ckks_copy(&mut input, &*ct, &mut scratch)?;
    input.set_log_delta(route.log_delta);
    trace_wide(module, &mut input, stage.trace_skip, route.extra_bits, keys, &mut scratch)?;

    module.ckks_copy(&mut x, &input, &mut scratch)?;
    x.set_log_delta(route.log_delta + route.extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    module.ckks_dft_matrix_eval_assign(&mut x, dft, keys, &mut scratch)?;
    x.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(x.log_delta()),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Full,
    });
    ct.set_k((x.log_budget() + route.log_delta).into());
    module.ckks_round(ct, &x, &mut scratch)
}

/// CoeffsToSlots of a replicated run, in place on `ct`, the raised ciphertext `extra_bits` under its scale: the
/// normalized trace onto period `k` removes the lifted multiples off the subring, and `dft` writes the coefficients
/// onto the data slots, of two batches side by side when packed. `ct` keeps its working bits for EvalMod to round.
pub(crate) fn coeffs_to_slots<BE, D, C, H>(
    module: &Module<BE>,
    ct: &mut C,
    dft: &D,
    stage: ReplicatedStage<'_>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSDFTOps<BE> + GLWETrace<BE>,
    D: CKKSDFTFactors<BE, Encode, Standard>,
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_coeffs_to_slots_assign";
    crate::ckks_ensure_operand!(
        OP,
        ct.slot_layout() == SlotLayout::Full,
        "a replicated run takes the raised ciphertext, of the full layout"
    );
    let log_n = ct.n().log2();
    // The trace of core looks its keys up and switches with them without a refusal of its own.
    let lookups = module
        .glwe_trace_galois_elements(ct.n(), stage.trace_skip..log_n, false)
        .into_iter()
        .map(|p| (p, ct.k().as_usize()));
    crate::layouts::validation::ensure_galois_keys(OP, keys, lookups)?;
    // The trace writes `ct`: the keys of the transform after it are checked first.
    let chain = crate::layouts::validation::chain_galois_lookups(
        OP,
        module,
        ct.n().as_usize(),
        dft.factor_plans(),
        stage.consumed,
        ct.k().as_usize(),
    )?;
    crate::layouts::validation::ensure_galois_keys(OP, keys, chain)?;
    module.glwe_trace_assign(ct, stage.trace_skip..log_n, false, true, keys, scratch);
    ct.set_log_sparsity(stage.top_sparsity);
    module.ckks_dft_matrix_eval_assign(ct, dft, keys, scratch)?;
    ct.set_slot_layout(SlotLayout::Replicated(stage.c2s_layout()));
    Ok(())
}

/// EvalMod of a replicated run on `a`, the output of CoeffsToSlots `extra_bits` under its scale, into `res`.
///
/// The real part `a + conj(a)` and, unpacked, the imaginary part `i·(conj(a) − a)`, are rounded with shaping to the
/// raised scale; EvalMod runs on each, or once on the two batches when packed, which one rotation by `k` values
/// recombines, and the parts are recombined into `Re + i·Im`. The result, `Δ_b·m` at the raised scale, is `m` at `Δ_b`
/// lifted by `mod_up_extra_bits`, and one rounding with shaping lands it at the leveled scale in `res`, whose width is
/// the output width of the run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_mod<BE, R, A, F, K, T>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    stage: ReplicatedStage<'_>,
    eval_mod: &EvalMod<F, MaskedCoeffs<CKKSPlaintextOwned<BE>>>,
    keys: &K,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>
        + CKKSRoundOps<BE>
        + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: GetAutomorphismKey<BE>,
    T: GetTensorKey<BE>,
{
    let route = stage.route;
    let raised_log_delta = a.log_delta() - route.extra_bits;
    let wide = a.glwe_layout();
    let (mut conj, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&wide, a.meta());
    let (mut part, scratch) = scratch.take_ckks_ciphertext_scratch(&wide, a.meta());
    let (mut re, scratch) = scratch.take_ckks_ciphertext_scratch(&wide, a.meta());
    let (mut im, scratch) = scratch.take_ckks_ciphertext_scratch(&wide, a.meta());
    let (mut out, mut scratch) = scratch.take_ckks_ciphertext_scratch(&wide, a.meta());

    module.ckks_conjugate_into(&mut conj, a, keys, &mut scratch)?;
    module.ckks_add_into(&mut part, a, &conj, &mut scratch)?;
    part.set_slots(SlotsKind::Real);
    re.set_k((part.log_budget() + raised_log_delta).into());
    module.ckks_round(&mut re, &part, &mut scratch)?;
    if stage.packed.is_none() {
        module.ckks_sub_into(&mut part, &conj, a, &mut scratch)?;
        module.ckks_mul_i_assign(&mut part, &mut scratch)?;
        part.set_slots(SlotsKind::Real);
        im.set_k((part.log_budget() + raised_log_delta).into());
        module.ckks_round(&mut im, &part, &mut scratch)?;
    }

    // EvalMod takes its input as working storage: `re` and `im` are not read again.
    module.ckks_eval_mod(&mut out, &mut re, eval_mod, tsk, &mut scratch)?;
    match stage.packed {
        Some(double) => {
            // The second batch came out times `i`: one rotation by `k` values adds it.
            let k = 1i64 << double.log_k();
            // At the width of `out`, which the entry check looks the key up at.
            re.set_k(out.k());
            module.ckks_rotate_into((k / 2) << double.log_r(), &mut re, &out, keys, &mut scratch)?;
            module.ckks_add_assign(&mut out, &re, &mut scratch)?;
            out.set_log_sparsity(GLWEPlaintextInfos::log_sparsity(&out) + 1);
        }
        None => {
            module.ckks_eval_mod(&mut re, &mut im, eval_mod, tsk, &mut scratch)?;
            module.ckks_mul_i_assign(&mut re, &mut scratch)?;
            module.ckks_add_assign(&mut out, &re, &mut scratch)?;
        }
    }
    out.set_slots(SlotsKind::Complex);
    out.set_slot_layout(SlotLayout::Replicated(route.layout));
    out.set_scale(Scale::Log(route.log_delta + route.mod_up_extra_bits));
    module.ckks_round(res, &out, &mut scratch)
}

/// What [`eval_mod`] refuses before it writes `res`: operands of rank other than one or of two degrees, an input off
/// the layout of CoeffsToSlots or without its working bits, and an output whose width does not land the result at the
/// leveled scale.
pub(crate) fn check_eval_mod<BE, R, A>(
    op: &'static str,
    module: &Module<BE>,
    res: &R,
    a: &A,
    stage: ReplicatedStage<'_>,
    eval_mod_consumed: usize,
) -> Result<()>
where
    BE: Backend,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
{
    let route = stage.route;
    crate::layouts::validation::ensure_rank_one(op, a, res)?;
    crate::layouts::validation::ensure_one_degree(op, module.n(), a, res)?;
    crate::ckks_ensure_operand!(
        op,
        a.slot_layout() == SlotLayout::Replicated(stage.c2s_layout()),
        "takes the output of the replicated CoeffsToSlots"
    );
    let raised_log_delta = a.log_delta().checked_sub(route.extra_bits);
    crate::ckks_ensure_operand!(
        op,
        raised_log_delta.is_some_and(|d| d >= route.log_delta + route.mod_up_extra_bits),
        "takes an input {} bits under the raised scale",
        route.extra_bits
    );
    let budget = crate::checked_log_budget_sub(op, a.log_budget(), eval_mod_consumed)?;
    // `Δ_b·m` at the raised scale, relabelled at `Δ_b` lifted by `mod_up_extra_bits`, at the leveled scale.
    let relabelled = budget + raised_log_delta.unwrap_or_default() - route.log_delta - route.mod_up_extra_bits;
    let k_out = relabelled + route.level_log_delta;
    crate::ckks_ensure_operand!(
        op,
        res.k().as_usize() == k_out,
        "lands the result at the leveled scale in an output of {k_out} bits, got {}",
        res.k().as_usize()
    );
    Ok(())
}

/// Scratch of [`eval_mod`]: its five working ciphertexts at the layout of `a`, beside the largest of its steps under
/// the keys of `keys_layout`.
pub(crate) fn eval_mod_tmp_bytes<BE, R, A, F>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    stage: ReplicatedStage<'_>,
    eval_mod: &EvalMod<F, MaskedCoeffs<CKKSPlaintextOwned<BE>>>,
    keys_layout: &crate::layouts::BootstrappingKeysLayout,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>
        + CKKSRoundOps<BE>
        + CKKSEvalModOps<BE>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
{
    let wide = crate::CKKSLayout {
        glwe_layout: a.glwe_layout(),
        meta: GLWEPlaintextMeta {
            slot_layout: SlotLayout::Replicated(stage.c2s_layout()),
            ..a.meta()
        },
    };
    let size = wide.glwe_layout.size();
    let mut ops = module
        .ckks_add_tmp_bytes(size)
        .max(module.ckks_sub_tmp_bytes(size))
        .max(module.ckks_mul_i_tmp_bytes(size))
        .max(module.ckks_round_tmp_bytes(&wide, &wide))
        .max(module.ckks_round_tmp_bytes(res, &wide))
        .max(module.ckks_eval_mod_tmp_bytes(&wide, &wide, eval_mod, &keys_layout.tensor_key));
    for key in keys_layout.rotation_key_layouts() {
        ops = ops
            .max(module.ckks_conjugate_tmp_bytes(&wide, key))
            .max(module.ckks_rotate_tmp_bytes(&wide, key));
    }
    5 * crate::oep::derived::replicated::ct_bytes::<BE>(&wide.glwe_layout) + ops
}

/// Scratch of the step of a replicated run on `ct`, with keys of layout `key`: SlotsToCoeffs, its input and working
/// ciphertexts beside the wide trace, the copies, `chain` of the working layout and the rounding; CoeffsToSlots, the
/// trace beside `chain` of `ct`.
pub(crate) fn stage_tmp_bytes<BE, K>(
    module: &Module<BE>,
    ct: &crate::CKKSLayout,
    stage: ReplicatedStage<'_>,
    key: &K,
    chain: impl Fn(&crate::CKKSLayout) -> usize,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE> + CKKSRoundOps<BE> + GLWETrace<BE>,
    K: poulpy_core::layouts::GGLWEInfos,
{
    let at = |k: usize| crate::CKKSLayout {
        glwe_layout: GLWELayout {
            k: k.into(),
            ..ct.glwe_layout
        },
        meta: ct.meta,
    };
    let ct_bytes = |layout: &crate::CKKSLayout| crate::oep::derived::replicated::ct_bytes::<BE>(&layout.glwe_layout);
    match stage.step {
        ReplicatedStep::SlotsToCoeffs => {
            let route = stage.route;
            let k_in = route.log_delta + stage.log_msg_ratio + route.s2c_consumed_bits();
            let (input, x) = (at(k_in), at(k_in + route.extra_bits));
            let trace = ct_bytes(&x)
                + module
                    .ckks_copy_tmp_bytes(&x, &input)
                    .max(module.glwe_trace_assign_tmp_bytes(&x.glwe_layout, key))
                    .max(module.ckks_round_tmp_bytes(&input, &x));
            ct_bytes(&input)
                + ct_bytes(&x)
                + module
                    .ckks_copy_tmp_bytes(&input, ct)
                    .max(trace)
                    .max(module.ckks_copy_tmp_bytes(&x, &input))
                    .max(chain(&x))
                    .max(module.ckks_round_tmp_bytes(ct, &x))
        }
        ReplicatedStep::CoeffsToSlots => module.glwe_trace_assign_tmp_bytes(&ct.glwe_layout, key).max(chain(ct)),
    }
}
