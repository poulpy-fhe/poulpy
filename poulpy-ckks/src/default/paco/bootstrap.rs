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
    Compact, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPreparedToBackendRef, GLWEInfos, GLWETensorKeyPreparedToBackendRef,
    GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena};

use super::ops::PaCoSlotOps;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSMeta,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSSubOps, PaCoScalar},
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext,
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
fn checked_output_meta<B: CKKSCtBounds>(plan: &PaCoPlan, input: &impl CKKSCtBounds, bsk: &B) -> Result<(usize, usize)> {
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

    let final_k = bsk
        .k()
        .as_usize()
        .checked_sub(plan.consumed_bits())
        .context("PaCo key width is smaller than the plan's budget consumption")?;
    ckks_ensure!(
        output_scale <= final_k,
        "PaCo output scale {output_scale} exceeds the post-bootstrap torus width {final_k}",
    );
    Ok((output_scale, final_k))
}

/// Validates the runtime ciphertexts and every key reachable through a custom
/// [`PaCoKeys`] implementation. Validated built-in key bundles satisfy the same
/// checks at construction; repeating the inexpensive shape checks here keeps
/// the public trait safe for lazy or external key managers.
pub(crate) fn validate_runtime<BE, F, K, Src>(
    module: &Module<BE>,
    context: &PaCoContext<BE, F>,
    output: &CKKSCiphertext<BE::OwnedBuf>,
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
    ckks_ensure!(
        input.log_sparsity() == 0,
        "PaCo input must be dense, got log_sparsity={}",
        input.log_sparsity()
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
    // The circuit runs entirely in-place on `output` at the bootstrapping-key
    // width, so its allocated limb capacity must equal that width exactly: a
    // narrower buffer cannot hold the result, and a wider one would drive the
    // in-place keyswitches past what the gadget keys can process.
    ckks_ensure!(
        output.max_size() == canonical.max_size(),
        "PaCo output must be allocated to exactly the bootstrapping-key width of {} limbs, got {} limbs",
        canonical.max_size(),
        output.max_size(),
    );
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

    // Automorphism and multiplication keyswitches size their working result from
    // the destination's allocated limb capacity, which the check above pins to the
    // bootstrapping-key width.
    let working_size = canonical.max_size();

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
        let key = keys
            .rotation_keys()
            .get_automorphism_key(element)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
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

    checked_output_meta(plan, input, canonical)
}

/// Evaluates one direct PaCo branch into caller-provided `output`.
///
/// The input is already encrypted under the structured PaCo secret. The four
/// bootstrapping ciphertexts homomorphically decrypt it into the application
/// key, so `output` is under the key that encrypts those ciphertexts. Runtime
/// validation is completed before the output is mutated.
pub(crate) fn paco_bootstrap_branch_validated_into<BE, F, K, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertext<BE::OwnedBuf>,
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
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + CyclotomicOrder,
    F: PaCoScalar,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + Compact,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let (output_scale, expected_final_k) = output_meta;
    let plan = context.plan();
    let bsk = keys.bootstrapping_keys();

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
    let mut temporary = module.ckks_ciphertext_alloc(context.base2k(), bsk[0].k());
    module.ckks_mul_pt_vec_into(output, &bsk[0], &beta[0], scratch)?;
    for index in 1..4 {
        module.ckks_mul_pt_vec_into(&mut temporary, &bsk[index], &beta[index], scratch)?;
        module.ckks_add_assign(output, &temporary, scratch)?;
    }

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
    let conjugation_key = keys
        .rotation_keys()
        .get_automorphism_key(-1)
        .ok_or(CKKSCompositionError::MissingAutomorphismKey {
            op: "ckks_paco_bootstrap",
            rotation: -1,
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
            let conj_rotate_key = keys.rotation_keys().get_automorphism_key(*galois_element).ok_or(
                CKKSCompositionError::MissingAutomorphismKey {
                    op: "ckks_paco_bootstrap",
                    rotation: *galois_element,
                },
            )?;
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
        keys.tensor_key(),
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
    // gap — not a correction, but the genuinely denser structure the recombined
    // sum of `kappa` interleaved branches then holds.
    let gap = plan.n() / plan.c();
    ckks_ensure!(
        gap.is_power_of_two(),
        "PaCo branch coefficient gap {gap} is not a power of two"
    );
    output.set_meta_checked(CKKSMeta {
        log_delta: output_scale,
        log_sparsity: gap.trailing_zeros() as usize,
    })?;
    Ok(())
}
