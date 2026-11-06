//! EvalMod on the conjugate-invariant ring: the conversions there and back around one EvalMod.

use poulpy_core::{
    GLWEBytesOf, GLWECIEmbed, GLWECITrace, GLWEKeyswitch, GLWELinearTransformationBabyStepsBytesOf, GLWERotate, GLWESwitchDegree,
    LinearTransformationPrepared, ScratchArenaTakeGLWELinearTransformationBabySteps,
    layouts::{
        GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformationPlan, SetK,
        prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSEvalModOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSPow2Ops, CKKSSubOps,
        ckks_plaintext_diagonal_layout,
    },
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, CKKSCITwin, CKKSEvalModInfos, EvalModCIPrepared,
        ScratchArenaTakeCKKSCiphertext,
        bootstrapping::{CIPreForm, bootstrap_layouts},
        validation::check_conversion_key_widths,
    },
    reference::linear_transformation::PlaintextDiagonalKeys,
};

use super::eval_mod::EVAL_MOD_OP;

/// Scratch of [`eval_mod_on_ci`] at the width of `boot_layout`, on a layout with conversion keys:
/// the degree-`2N` ciphertext beside the conversions, the traced and the reduced ciphertext
/// beside EvalMod.
pub(super) fn eval_mod_on_ci_tmp_bytes<BE, F>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    boot_layout: &CKKSLayout,
    eval_mod_ci: &EvalModCIPrepared<BE, F>,
    eval_round_plus: bool,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSCITwin,
    Module<BE>: GLWEBytesOf<BE>
        + GLWELinearTransformationBabyStepsBytesOf<BE>
        + CKKSPow2Ops<BE>
        + GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWERotate<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
{
    // A layout without conversions fails at the bootstrap.
    let Some((trace_key, embed_key)) = &keys_layout.conversion_keys else {
        return 0;
    };
    let embedded = CKKSLayout {
        glwe_layout: GLWELayout {
            n: (2 * boot_layout.n().as_usize()).into(),
            ..boot_layout.glwe_layout
        },
        ..*boot_layout
    };
    let embedded_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&embedded));
    let trace = embedded_bytes
        + module
            .glwe_keyswitch_tmp_bytes(&embedded, &embedded, trace_key)
            .max(module.glwe_ci_trace_tmp_bytes(boot_layout, &embedded));
    let pre = match &eval_mod_ci.pre.0 {
        CIPreForm::Product(u) => module.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, u),
        // The baby steps of `even` and the outputs of the two folded factors beside
        // their evaluation, then the odd one's embedding beside its rotation.
        CIPreForm::Folded { plans, .. } => {
            // The baby steps of `even`, which `odd` shares, and each factor on its own schedule,
            // which counts the doubling and the giant-step tree of power-of-two keys.
            let [even_plan, odd_plan] = &**plans;
            let diagonal = ckks_plaintext_diagonal_layout(boot_layout);
            let mut factors = 0;
            for atk in keys_layout.rotation_key_layouts() {
                factors = factors
                    .max(module.ckks_linear_transformation_baby_steps_prepare_tmp_bytes(boot_layout, even_plan, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, even_plan, atk, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, odd_plan, atk, atk));
            }
            module.glwe_linear_transformation_baby_steps_bytes_of(even_plan, boot_layout)
                + 2 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(boot_layout))
                + factors.max(embedded_bytes + module.glwe_rotate_tmp_bytes().max(module.ckks_add_tmp_bytes(embedded.size())))
        }
    };
    let conversions = pre
        .max(module.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, &eval_mod_ci.post))
        .max(module.glwe_keyswitch_tmp_bytes(&embedded, &embedded, embed_key))
        .max(module.ckks_mul_pow2_tmp_bytes(boot_layout.size()))
        .max(module.ckks_sub_tmp_bytes(boot_layout.size()))
        .max(trace);
    let eval_mod = ci.ckks_eval_mod_tmp_bytes(boot_layout, boot_layout, &eval_mod_ci.eval_mod, &keys_layout.tensor_key);
    let ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(boot_layout));
    embedded_bytes + (2 + usize::from(eval_round_plus)) * ct_bytes + conversions.max(eval_mod)
}

/// `u·z(X²)` into `embedded`, of degree `2N`, from the input `w` of the last
/// CoeffsToSlots factor folded with `u` ([`CIPrePrepared::Folded`]): `a(X²) + X·b(X²)`,
/// with `a` and `b` the even and odd factors on `w`, which share its baby steps.
fn ci_folded_pre<BE, R, W, H>(
    module: &Module<BE>,
    embedded: &mut R,
    w: &W,
    [even, odd]: [&LinearTransformationPrepared<BE>; 2],
    plan: &LinearTransformationPlan,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + GLWERotate<BE> + CKKSAddOps<BE> + GLWESwitchDegree<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    W: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let (mut babies, mut scratch) = scratch
        .borrow()
        .take_glwe_linear_transformation_baby_steps_scratch(module, plan, w);
    module.ckks_linear_transformation_baby_steps_prepare(&mut babies, w, keys, &mut scratch)?;
    let (mut a, scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(w, w.meta());
    let (mut b, mut scratch_1) = scratch_1.take_ckks_ciphertext_scratch(w, w.meta());
    let lt_keys = PlaintextDiagonalKeys(keys);
    module.ckks_linear_transformation_eval_into(&mut a, w, &babies, even, &lt_keys, &mut scratch_1)?;
    module.ckks_linear_transformation_eval_into(&mut b, w, &babies, odd, &lt_keys, &mut scratch_1)?;
    let (mut odd_wide, mut scratch_2) = scratch_1.take_ckks_ciphertext_scratch(embedded, b.meta());
    odd_wide.set_k(b.k());
    module.glwe_switch_degree(&mut odd_wide, &b);
    // `X·b(X²)` fills the odd coefficients, which `a(X²)` leaves at zero.
    module.glwe_rotate_assign(1, &mut odd_wide, &mut scratch_2);
    embedded.set_meta(a.meta());
    embedded.set_k(a.k());
    module.glwe_switch_degree(embedded, &a);
    module.ckks_add_assign(embedded, &odd_wide, &mut scratch_2)?;
    Ok(())
}

/// One EvalMod for all `N` coefficients, on the conjugate-invariant ring: `a` to degree `2N`, its
/// product by `u`, the trace onto the CI ring, EvalMod on `ci`, and back by the embedding and `v`.
/// The two rings share their buffers, so `module` and `ci` work on one arena.
#[expect(
    clippy::too_many_arguments,
    reason = "the route reads a module per ring and the CI part of the plan"
)]
pub(super) fn eval_mod_on_ci<BE, F, R, A, P, K>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    res: &mut R,
    a: &A,
    plan: &P,
    eval_mod_ci: &EvalModCIPrepared<BE, F>,
    (tensor_key, trace_key, embed_key): (&K::CITensorKey, &K::TraceKey, &K::EmbedKey),
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    Module<BE>: GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + GLWECIEmbed<BE>
        + GLWERotate<BE>
        + GLWESwitchDegree<BE>
        + CKKSSubOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
    K: BootstrappingKeys<BE>,
{
    let boot_layout = bootstrap_layouts(&*res, a).0.glwe_layout;
    let wide_layout = GLWELayout {
        n: (2 * boot_layout.n.as_usize()).into(),
        ..boot_layout
    };
    let (mut embedded, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&wide_layout, a.meta());

    // `z(X²)` times `u`, traced onto the CI ring without halving: `2·Re(u·z)`, the
    // split EvalMod input.
    match &eval_mod_ci.pre.0 {
        CIPreForm::Product(u) => {
            embedded.set_k(a.k());
            module.glwe_switch_degree(&mut embedded, a);
            module.ckks_mul_pt_vec_assign(&mut embedded, u, &mut scratch)?;
        }
        CIPreForm::Folded { even, odd, plans } => {
            ci_folded_pre(module, &mut embedded, a, [even, odd], &plans[0], keys, &mut scratch)?
        }
    }
    let widths = plan.eval_mod_plan().conversion_widths(embedded.k().as_usize());
    check_conversion_key_widths(EVAL_MOD_OP, trace_key, embed_key, widths)?;
    let meta = GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..embedded.meta()
    };
    let (mut traced, scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, meta);
    let (reduced, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, meta);
    traced.set_k(embedded.k());
    {
        let (mut switched, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&embedded, embedded.meta());
        module.glwe_keyswitch(&mut switched, &embedded, &trace_key.to_backend_ref(), &mut scratch);
        module.glwe_ci_trace(&mut traced, &switched, &mut scratch);
    }

    // EvalMod works on `traced`, so `K·x` comes first. It is coefficient-wise, the same on either ring.
    let (scaled, mut scratch) = if plan.eval_round_plus() {
        let log2_k = plan.eval_mod_plan().f_mod_interval.trailing_zeros() as usize;
        let (mut scaled, mut rest) = scratch.take_ckks_ciphertext_scratch(&boot_layout, traced.meta());
        scaled.set_k(traced.k());
        module.ckks_mul_pow2_into(log2_k, &mut scaled, &traced, &mut rest)?;
        (Some(scaled), rest)
    } else {
        (None, scratch)
    };

    let mut traced = traced.rebind::<BE::CI>();
    let mut reduced = reduced.rebind::<BE::CI>();
    reduced.set_k(traced.k());
    ci.ckks_eval_mod(
        &mut reduced,
        &mut traced,
        &eval_mod_ci.eval_mod,
        tensor_key,
        &mut scratch.borrow().into_backend::<BE::CI>(),
    )?;
    let mut reduced = reduced.rebind::<BE>();
    if let Some(scaled) = &scaled {
        module.ckks_sub_assign(&mut reduced, scaled, &mut scratch)?;
    }

    // Back to degree `2N` and times `v`, which carries the factor 2 of the
    // conjugate pair: `r_j + i·r_{j+N/2}` on the even coefficients, under `s(X²)`.
    embedded.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..reduced.meta()
    });
    embedded.set_k(reduced.k());
    module.glwe_ci_embed(&mut embedded, &reduced);
    module.glwe_keyswitch_assign(&mut embedded, &embed_key.to_backend_ref(), &mut scratch);
    module.ckks_mul_pt_vec_assign(&mut embedded, &eval_mod_ci.post, &mut scratch)?;
    res.set_k(embedded.k());
    module.glwe_switch_degree(res, &embedded);
    res.set_meta(embedded.meta());
    Ok(())
}
