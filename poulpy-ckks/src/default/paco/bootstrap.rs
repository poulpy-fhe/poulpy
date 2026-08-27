//! Backend-generic sequential PaCo branch circuit and runtime validation.
//!
//! The compiled evaluation material lives in
//! [`PaCoContext`](crate::layouts::PaCoContext); this module owns the
//! input-dependent work: preflight validation of the runtime ciphertexts and
//! keys, and the seqPaCo branch evaluation the sequential and parallel
//! drivers share.

use crate::{CKKSResult as Result, ckks_ensure};
use anyhow::Context;
use poulpy_core::layouts::{
    GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPreparedToBackendRef, GLWEInfos, GLWETensorKeyPreparedToBackendRef,
    GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena};

use super::ops::PaCoSlotOps;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSMeta,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSSubOps, PaCoScalar},
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned,
        paco::{
            context::{PaCoContext, PaCoPsiTailMaterial},
            keyset::{PaCoKeyParameters, PaCoKeys, validate_backend_storage_capacity, validate_gadget_backend_view},
            plan::PaCoPlan,
        },
    },
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl},
};

/// Metadata predicted for a direct PaCo branch after the final, budget-neutral
/// scale relabel. Computing it before evaluation prevents an expensive circuit
/// from ending in scale or budget underflow.
fn checked_output_meta<B: CKKSCtBounds>(
    plan: &PaCoPlan,
    input: &impl CKKSCtBounds,
    output_k: usize,
    bsk: &B,
) -> Result<(usize, usize)> {
    let input_scale = i128::try_from(input.log_delta()).context("PaCo input scale does not fit signed arithmetic")?;
    let bootstrap_scale = i128::try_from(bsk.log_delta()).context("PaCo bootstrap scale does not fit signed arithmetic")?;
    let shift = i128::from(plan.log_q())
        .checked_sub(2)
        .and_then(|value| value.checked_sub(input_scale))
        .and_then(|value| value.checked_sub(i128::from(plan.extra_scale_log2())))
        .context("PaCo output scale shift overflows signed arithmetic")?;
    let output_scale = bootstrap_scale
        .checked_sub(shift)
        .context("PaCo output scale overflows signed arithmetic")?;
    ckks_ensure!(output_scale >= 0, "PaCo output scale would be negative ({output_scale})");
    let output_scale = usize::try_from(output_scale).context("PaCo output scale does not fit usize")?;

    // `k_out` is the maximum final output level: the bootstrapping-key (seed) width
    // minus the budget the circuit consumes. The caller may request any output level
    // up to `k_out`; the branch then evaluates at working level `output_k + circuit_depth`
    // (< the seed width when `output_k < k_out`), so a lower level runs the whole circuit
    // narrower — genuinely cheaper.
    let k_out = bsk
        .k()
        .as_usize()
        .checked_sub(plan.consumed_bits())
        .context("PaCo key width is smaller than the plan's budget consumption")?;
    ckks_ensure!(
        output_k <= k_out,
        "PaCo output level {output_k} exceeds the maximum bootstrap output level {k_out}",
    );
    ckks_ensure!(
        output_scale <= output_k,
        "PaCo output scale {output_scale} exceeds the requested output width {output_k}",
    );
    Ok((output_scale, output_k))
}

/// Working torus width the branch evaluates at for a requested `output_k`.
///
/// The blind-rotation output `bsk·β` naturally lands at `bsk.k() - log_delta_bsk`, and
/// the DFT/product stages consume `consumed_bits - log_delta_bsk` more (`circuit_depth`).
/// Allocating the working accumulator at `output_k + circuit_depth` makes the first
/// multiply produce the phase directly at that width (a value-preserving rounding of the
/// low bits, since ct×pt targets `dst.k()`), so the fixed-depth circuit lands exactly on
/// `output_k`. At `output_k == k_out` this equals the phase's natural width (no rounding);
/// below it the whole circuit runs narrower — genuinely cheaper.
pub(crate) fn branch_working_k(plan: &PaCoPlan, output_k: usize) -> Result<usize> {
    let circuit_depth = plan
        .consumed_bits()
        .checked_sub(plan.log_delta_bsk())
        .context("PaCo consumed budget is smaller than the seed rescale")?;
    output_k
        .checked_add(circuit_depth)
        .context("PaCo working torus width overflows usize")
        .map_err(Into::into)
}

/// Validates the runtime ciphertexts and every key reachable through a custom
/// [`PaCoKeys`] implementation. Validated built-in key bundles satisfy the same
/// checks at construction; repeating the inexpensive shape checks here keeps
/// the public trait safe for lazy or external key managers.
pub(crate) fn validate_runtime<BE, F, K, Src>(
    module: &Module<BE>,
    context: &PaCoContext<BE, F>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &K,
) -> Result<(usize, usize)>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE>,
    F: PaCoScalar,
    Module<BE>: CyclotomicOrder,
    K: PaCoKeys<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let plan = context.plan();
    ckks_ensure!(
        keys.parameters() == PaCoKeyParameters::from_plan(plan),
        "PaCo key parameters {:?} do not match context plan {:?}",
        keys.parameters(),
        PaCoKeyParameters::from_plan(plan),
    );
    ckks_ensure!(
        module.n() == plan.n(),
        "PaCo module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ckks_ensure!(
        input.n().as_usize() == plan.n(),
        "PaCo input degree {} does not match plan degree {}",
        input.n(),
        plan.n()
    );
    ckks_ensure!(
        input.rank().as_usize() == 1,
        "PaCo input must have rank 1, got {}",
        input.rank()
    );
    ckks_ensure!(
        input.k().as_usize() == plan.log_q() as usize,
        "PaCo input torus width {} does not match plan log_q {}",
        input.k(),
        plan.log_q(),
    );
    ckks_ensure!(
        input.log_delta() <= input.k().as_usize(),
        "PaCo input scale {} exceeds its torus width {}",
        input.log_delta(),
        input.k()
    );
    validate_backend_storage_capacity::<BE, _>("PaCo input", input)?;

    ckks_ensure!(
        output.n().as_usize() == plan.n(),
        "PaCo output degree {} does not match plan degree {}",
        output.n(),
        plan.n()
    );
    ckks_ensure!(
        output.rank().as_usize() == 1,
        "PaCo output must have rank 1, got {}",
        output.rank()
    );
    ckks_ensure!(
        output.base2k() == context.base2k(),
        "PaCo output base2k {} does not match context base2k {}",
        output.base2k(),
        context.base2k()
    );

    validate_backend_storage_capacity::<BE, _>("PaCo output", output)?;
    let bsk = keys.bootstrapping_keys();
    let canonical = &bsk[0];
    ckks_ensure!(
        canonical.k().as_usize() >= plan.max_plaintext_width(),
        "PaCo bootstrapping-key width {} is smaller than the widest plaintext width {}",
        canonical.k(),
        plan.max_plaintext_width(),
    );
    // The caller allocates `output` at the final level it wants, up to `k_out`; the
    // circuit runs at working level `output.k() + consumed_bits` in scratch (see the
    // branch executor). This both validates `output.k() <= k_out` and yields the
    // output metadata reused below.
    let output_meta = checked_output_meta(plan, input, output.k().as_usize(), canonical)?;
    for (index, key) in bsk.iter().enumerate() {
        ckks_ensure!(
            key.n().as_usize() == plan.n(),
            "PaCo bootstrapping key {index} has incompatible degree {}",
            key.n()
        );
        ckks_ensure!(
            key.rank().as_usize() == 1,
            "PaCo bootstrapping key {index} must have rank 1, got {}",
            key.rank()
        );
        ckks_ensure!(
            key.base2k() == context.base2k(),
            "PaCo bootstrapping key {index} has incompatible base2k {}",
            key.base2k()
        );
        ckks_ensure!(
            key.k() == canonical.k(),
            "PaCo bootstrapping key {index} has torus width {}, expected {}",
            key.k(),
            canonical.k()
        );
        ckks_ensure!(
            key.log_delta() == plan.log_delta_bsk(),
            "PaCo bootstrapping key {index} has scale {}, expected {}",
            key.log_delta(),
            plan.log_delta_bsk()
        );
        ckks_ensure!(
            key.log_delta() <= key.k().as_usize(),
            "PaCo bootstrapping key {index} scale {} exceeds torus width {}",
            key.log_delta(),
            key.k(),
        );
        validate_backend_storage_capacity::<BE, _>(&format!("PaCo bootstrapping key {index}"), key)?;
        ckks_ensure!(key.log_sparsity() == 0, "PaCo bootstrapping key {index} must be dense");
        ckks_ensure!(
            key.log_budget() >= plan.consumed_bits(),
            "PaCo bootstrapping key {index} has {} budget bits; the plan consumes {}",
            key.log_budget(),
            plan.consumed_bits()
        );
    }

    // Automorphism and multiplication keyswitches size their working result from the
    // working accumulator's limb capacity (`branch_working_k`). `output.k() <= k_out`
    // keeps this within the gadget keys' capacity, but re-validate defensively (custom
    // key managers may under-size).
    let working_size = branch_working_k(plan, output_meta.1)?.div_ceil(context.base2k().as_usize());

    let tensor_view = GLWETensorKeyPreparedToBackendRef::to_backend_ref(keys.tensor_key());
    validate_gadget_backend_view(
        "PaCo tensor key",
        keys.tensor_key(),
        &tensor_view,
        plan.n(),
        context.base2k(),
        working_size,
    )?;

    for &element in context.galois_elements() {
        let (key, _) = keys
            .rotation_keys()
            .get_automorphism_key_for(element, output.k())
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "ckks_paco_bootstrap",
                rotation: element,
            })?;
        ckks_ensure!(
            key.p() == element,
            "PaCo rotation-key map returned Galois element {} for label {element}",
            key.p()
        );
        let key_view = GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(key);
        ckks_ensure!(
            key_view.p() == element,
            "PaCo automorphism-key backend view returned Galois element {} for label {element}",
            key_view.p(),
        );
        validate_gadget_backend_view(
            "PaCo automorphism key",
            key,
            &key_view,
            plan.n(),
            context.base2k(),
            working_size,
        )?;
    }

    Ok(output_meta)
}

/// Evaluates one direct PaCo branch into caller-provided `output`.
///
/// The input is already encrypted under the structured PaCo secret. The four
/// bootstrapping ciphertexts homomorphically decrypt it into the application
/// key, so `output` is under the key that encrypts those ciphertexts. Runtime
/// validation is completed before the output is mutated.
pub(crate) fn paco_bootstrap_branch_validated_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    context: &PaCoContext<BE, F>,
    keys: &K,
    output_meta: (usize, usize),
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    K: PaCoKeys<BE>,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + CyclotomicOrder,
    F: PaCoScalar,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let (output_scale, expected_final_k) = output_meta;
    let plan = context.plan();
    let bsk = keys.bootstrapping_keys();

    // The circuit runs in a dedicated accumulator sized to `output_k + circuit_depth`
    // (see `branch_working_k`), NOT in the caller's `output`. This truncates the seed
    // to the width whose post-stage result is exactly `expected_final_k`, so a lower
    // requested output level runs the entire circuit narrower — genuinely cheaper. The
    // result is copied into `output` at the end. `output` is shadowed below so the
    // circuit body operates on the working accumulator unchanged.
    let working_k = branch_working_k(plan, expected_final_k)?;
    let mut work = module.ckks_ciphertext_alloc(context.base2k(), TorusPrecision(working_k as u32));
    let destination = output;
    let output = &mut work;

    // Step 1 — Coefficient encoding. Turn the input ciphertext's public
    // coefficients into the four input-dependent β plaintexts the blind
    // rotation consumes (each β slot on the unit circle at scale
    // `2^log_delta_bsk`). Runs in a nested scope so the β scratch is released
    // before the circuit proper.
    let beta = scratch.scope(|arena| {
        let mut arena = arena;
        BE::ckks_paco_coeff_encodings_impl::<F, _>(module, input, context.plan(), context.base2k(), &mut arena)
    })?;

    // Step 2 — Blind rotation. Assemble the encrypted phase as the inner
    // product `Σ_t Enc(σ_t)·β_t` over the four structured-secret ciphertexts:
    // `bsk[0]·β[0]` seeds `output`, and each remaining `bsk[t]·β[t]`
    // accumulates through `temporary`.
    let mut temporary = module.ckks_ciphertext_alloc(context.base2k(), TorusPrecision(working_k as u32));
    module.ckks_mul_pt_vec_into(output, &bsk[0], &beta[0], scratch)?;
    for index in 1..4 {
        module.ckks_mul_pt_vec_into(&mut temporary, &bsk[index], &beta[index], scratch)?;
        module.ckks_add_assign(output, &temporary, scratch)?;
    }
    // The blind-rotation multiplies target `output.k()` (= working_k), so the phase is
    // produced directly at the working width with proper rounding — no relabel needed.

    // Step 3 — Homomorphic trace `Tr_{N/2 → slots}`. Collapse the redundant
    // coefficient-class copies of the phase into the working slot layout. One
    // fused automorphism-add per level; consumes no budget.
    module.ckks_slot_trace_assign(output, plan.half_n(), plan.slots(), keys.rotation_keys(), scratch)?;

    // Step 4 — CoeffsToSlots (body). Apply the compiled C2S factor chain (one
    // linear transformation per factor matrix), moving the packed coefficients
    // onto slots.
    for factor in context.coeffs_to_slots() {
        module.ckks_eval_linear_transformation_self_assign(output, factor, keys.rotation_keys(), scratch)?;
    }

    // Fetched once and reused by both the ψ tail (Pair form) and the
    // imaginary-part extraction (step 7): `conj(·)` via the order `-1`
    // automorphism key.
    let (conjugation_key, _) = keys.rotation_keys().get_automorphism_key_for(-1, output.k()).map_err(|_| {
        CKKSCompositionError::MissingAutomorphismKey {
            op: "ckks_paco_bootstrap",
            rotation: -1,
        }
    })?;

    // Step 5 — CoeffsToSlots (ψ tail). The conjugation-augmented final C2S
    // factor, scheduled apart from the body. Either the fused `Pair` form
    // `A·w + B·conj(w)` (one conjugation keyswitch, two transforms, one add) or
    // the operation-lean `Mask` form (one fused conj-rotate keyswitch, one add,
    // one multiply by the share-scaled μ mask).
    match context.psi_tail() {
        PaCoPsiTailMaterial::Pair(pair) => {
            module.ckks_conjugate_into(&mut temporary, output, conjugation_key, scratch)?;
            module.ckks_eval_linear_transformation_self_assign(output, &pair[0], keys.rotation_keys(), scratch)?;
            module.ckks_eval_linear_transformation_self_assign(&mut temporary, &pair[1], keys.rotation_keys(), scratch)?;
            module.ckks_add_assign(output, &temporary, scratch)?;
        }
        PaCoPsiTailMaterial::Mask { mu, galois_element } => {
            let (conj_rotate_key, _) = keys
                .rotation_keys()
                .get_automorphism_key_for(*galois_element, output.k())
                .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                    op: "ckks_paco_bootstrap",
                    rotation: *galois_element,
                })?;
            module.ckks_conjugate_into(&mut temporary, output, conj_rotate_key, scratch)?;
            module.ckks_add_assign(output, &temporary, scratch)?;
            module.ckks_mul_pt_vec_assign(output, mu, scratch)?;
        }
    }

    // Step 6 — Product fold `Pr_{slots → 2C}`. The rotate-and-multiply slot
    // reduction (`log(slots/2C)` ct×ct products, relinearized via the tensor
    // key) that multiplies the per-class phases together — the core nonlinear
    // stage of the branch. Consumes `log(slots/2C)·log_delta` budget bits.
    module.ckks_slot_product_assign(
        output,
        plan.slots(),
        plan.c().checked_mul(2).context("PaCo product target overflows usize")?,
        keys.rotation_keys(),
        keys.relinearization_keys(),
        scratch,
    )?;

    // Step 7 — Imaginary-part extraction. `output ← output − conj(output) =
    // 2i·Im(output)`, isolating the imaginary component that carries the
    // recovered coefficients.
    module.ckks_conjugate_into(&mut temporary, output, conjugation_key, scratch)?;
    module.ckks_sub_assign(output, &temporary, scratch)?;

    // Step 8 — SlotsToCoeffs′. Apply the compiled S2C factor chain, moving the
    // recovered slot values back to coefficient positions.
    for factor in context.slots_to_coeffs() {
        module.ckks_eval_linear_transformation_self_assign(output, factor, keys.rotation_keys(), scratch)?;
    }

    // Step 9 — Scale relabel (budget-neutral). Verify the evaluated torus width
    // matches the width predicted by `checked_output_meta`, then stamp the
    // branch's output scale.
    ckks_ensure!(
        output.k().as_usize() == expected_final_k,
        "PaCo internal budget mismatch: evaluation produced torus width {}, expected {expected_final_k}",
        output.k(),
    );
    // A branch recovers `C` coefficient classes packed at gap `N/C` from
    // position 0: the input is pre-rotated (in `run_branch_into`) so this
    // branch's class lands on coefficient 0, and the class offset is
    // re-applied only after this function returns. So the ciphertext produced
    // here genuinely carries sparsity `log2(N/C)`, and that is its metadata.
    // When `kappa > 1`, the driver later re-stamps the finer `log2(N/(kappa*C))`
    // gap, which is not a correction but the genuinely denser structure the
    // recombined sum of `kappa` interleaved branches then holds: for the
    // derived `kappa = N/(C*2^s)` that is the input's own `s`.
    let gap = plan.n() / plan.c();
    ckks_ensure!(
        gap.is_power_of_two(),
        "PaCo branch coefficient gap {gap} is not a power of two"
    );
    output.set_meta_checked(CKKSMeta {
        log_delta: output_scale,
        log_sparsity: gap.trailing_zeros() as usize,
        slots: input.slots(),
    })?;

    // Write the result into the caller's destination. The accumulator already carries
    // exactly `expected_final_k` (= the requested output level), so the copy is lossless.
    module.ckks_copy(destination, &*output, scratch)?;
    Ok(())
}
