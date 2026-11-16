use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECIEmbed, GLWECITrace, GLWECopy, GLWEKeyswitch, GLWEShift, GLWETrace,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWELayout,
        GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, Scale, SetK,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::oep::derived::replicated::{
    apply_transform, check_destination, check_module_degree, check_workspace, ct_bytes, invalid, transform_tmp_bytes,
    working_layout,
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextInfos, ReplicatedLayout, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout,
    SlotsKind,
    api::{
        CKKSAddOps, CKKSCIRingMapOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSPow2Ops, CKKSReplicatedBootstrapOps, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        BootstrappingKeys, CKKSCiphertextViewMut, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedTransform,
        ReplicatedTransformWorkspace, ScratchArenaTakeCKKSCiphertext,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
    reference::bootstrapping::mod_raise_body,
};

/// The rotation keys of a bootstrapping key set, which the replicated maps read: their diagonals are plaintexts, so
/// they never ask for a tensor key, as a map of automorphism keys answers.
struct RotationKeys<'a, K>(&'a K);

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for RotationKeys<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: poulpy_core::layouts::TorusPrecision,
    ) -> poulpy_core::Result<poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.0.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend, K> GetTensorKey<BE> for RotationKeys<'_, K> {
    fn lookup_tensor_key(
        &self,
        _k: poulpy_core::layouts::TorusPrecision,
    ) -> poulpy_core::Result<poulpy_core::layouts::prepared::GLWETensorKeyPreparedBackendRef<'_, BE>> {
        Err(poulpy_core::CoreError::GGLWEKeyUse {
            op: "get_tensor_key",
            detail: "the rotation keys of a bootstrapping hold no tensor key".into(),
        })
    }
}

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
    module.ckks_round(ct, &wide, &mut scratch)
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
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
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
    module.ckks_round(re, &part, &mut scratch)?;
    if let Some(im) = im {
        module.ckks_sub_into(&mut part, &conj, &w, &mut scratch)?;
        module.ckks_mul_i_assign(&mut part, &mut scratch)?;
        part.set_slots(SlotsKind::Real);
        im.set_k((part.log_budget() + src.log_delta()).into());
        module.ckks_round(im, &part, &mut scratch)?;
    }
    Ok(())
}

/// Shaping depths of the two ring maps of the conjugate-invariant bootstrapping
/// for an input at scale `2^log_delta_in` and an output at the leveled scale.
fn ci_bootstrap_depths(layout: &ReplicatedLayout, log_delta_in: usize, level_log_delta: usize) -> (usize, usize) {
    let profile = crate::shaping::ShapingProfile::new(layout);
    (profile.shaping_depth(log_delta_in), profile.shaping_depth(level_log_delta))
}

/// Reference of the [`CKKSReplicatedBootstrapImpl`](crate::oep::CKKSReplicatedBootstrapImpl) family,
/// selected by [`impl_ckks_replicated_bootstrap_reference_full!`](crate::impl_ckks_replicated_bootstrap_reference_full):
/// the bootstrapping of [`CKKSReplicatedBootstrapOps`] and the conjugate-invariant bootstrapping of
/// [`CKKSReplicatedCIBootstrapOps`](crate::api::CKKSReplicatedCIBootstrapOps).
pub trait CKKSReplicatedBootstrapReference<BE: Backend> {
    /// Scratch of [`Self::ckks_replicated_bootstrap_reference`].
    fn ckks_replicated_bootstrap_tmp_bytes_reference<C, I, F, A, T, S>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Self: CKKSLinearTransformationOps<BE>
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
        S: GGLWEInfos;

    /// [`CKKSReplicatedBootstrapOps::ckks_replicated_bootstrap`].
    fn ckks_replicated_bootstrap_reference<Dst, Src, F, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Self: CKKSLinearTransformationOps<BE>
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
        K: BootstrappingKeys<BE>;
}

impl<BE: Backend> CKKSReplicatedBootstrapReference<BE> for Module<BE> {
    fn ckks_replicated_bootstrap_tmp_bytes_reference<C, I, F, A, T, S>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Self: CKKSLinearTransformationOps<BE>
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
        let module = self;
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
        let wide = |layout: GLWELayout, transform: &ReplicatedTransform<BE>| {
            [replicated, SlotLayout::Full]
                .into_iter()
                .map(|slot_layout| {
                    let wide = ckks(layout, slot_layout);
                    transform_tmp_bytes(module, &wide, atk_infos, transform)
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
        let front = ct_bytes::<BE>(&input) + ct_bytes::<BE>(&narrow) + 3 * ct_bytes::<BE>(&low) + wide(low, &ctx.s2c).max(mod_up);
        let front = front.max(ct_bytes::<BE>(&input) + ct_bytes::<BE>(&low) + wide(low, &ctx.s2c));
        let back = 3 * ct_bytes::<BE>(&raised) + (4 * ct_bytes::<BE>(&high) + wide(high, &ctx.c2s)).max(eval_mod);
        ct_bytes::<BE>(&raised) + front.max(back)
    }

    fn ckks_replicated_bootstrap_reference<Dst, Src, F, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Self: CKKSLinearTransformationOps<BE>
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
        let module = self;
        const OP: &str = "ckks_replicated_bootstrap";
        check_destination(OP, dst, src)?;
        check_module_degree(OP, module.n(), src)?;
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
        // The bootstrapping keys are the automorphism key source, and EvalMod runs on the standard ring.
        let rotation_keys = &RotationKeys(keys);
        let tsk = crate::layouts::validation::required_tensor_key(OP, keys.tensor_key())?;
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
            module.ckks_round(&mut coeffs, &acc, &mut scratch)?;
            if coeffs.log_budget() != plan.eval_mod.log_msg_ratio {
                return Err(invalid(OP, "the coefficient message does not have the EvalMod message ratio"));
            }

            // 4. ModUp: `I·q_b + Δ_b·m` at the scale `q_b`, lifted.
            let log_q_b = coeffs.k().as_usize();
            raised.set_k(k_boot.into());
            mod_raise_body(
                module,
                &mut raised,
                &mut coeffs,
                0,
                plan.mod_up_extra_bits,
                None,
                0,
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
        // EvalMod takes its input as working storage: `re` and `im` are not read again.
        module.ckks_eval_mod(&mut out, &mut re, &ctx.eval_mod, tsk, &mut scratch)?;
        if let Some(double) = ctx.packed {
            // The second batch came out times `i`: one rotation by `k` values adds it.
            let k = 1i64 << double.log_k();
            module.ckks_rotate_into((k / 2) << double.log_r(), &mut re, &out, rotation_keys, &mut scratch)?;
            module.ckks_add_assign(&mut out, &re, &mut scratch)?;
            out.set_log_sparsity(GLWEPlaintextInfos::log_sparsity(&out) + 1);
        } else {
            module.ckks_eval_mod(&mut re, &mut im, &ctx.eval_mod, tsk, &mut scratch)?;
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
        module.ckks_round(dst, &out, &mut scratch)
    }
}

/// Reference of the [`CKKSReplicatedCIBootstrapImpl`](crate::oep::CKKSReplicatedCIBootstrapImpl) family,
/// selected by [`impl_ckks_replicated_ci_bootstrap_reference_full!`](crate::impl_ckks_replicated_ci_bootstrap_reference_full):
/// the bootstrapping of [`CKKSReplicatedCIBootstrapOps`](crate::api::CKKSReplicatedCIBootstrapOps), whose
/// conjugate-invariant input and output of degree `N` live on the module of degree `2N`.
pub trait CKKSReplicatedCIBootstrapReference<BE: Backend> {
    /// Scratch of [`Self::ckks_replicated_ci_bootstrap_reference`]: the working ciphertexts of both rings
    /// beside the largest operation.
    #[allow(clippy::too_many_arguments)]
    fn ckks_replicated_ci_bootstrap_tmp_bytes_reference<C, I, F, A, T, S, K>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> usize
    where
        Self: CKKSReplicatedBootstrapOps<BE>
            + CKKSCIRingMapOps<BE>
            + CKKSCopyOps<BE>
            + CKKSRoundOps<BE>
            + CKKSCopyOps<BE, ConjugateInvariant>
            + CKKSRoundOps<BE, ConjugateInvariant>,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos;

    /// Bootstraps the conjugate-invariant replicated ciphertext `src` of degree
    /// `N` into `dst` through the standard ring of degree `2N` of `self`, where
    /// `ctx` was compiled: `src` is embedded, the standard bootstrapping refreshes
    /// it, and the relative trace brings it back. Each ring map runs at the
    /// shaping depth under the scale and rounds with shaping, so their key
    /// switches and the trace's halving stay off the data slots. `dst` ends at
    /// its own width and the leveled scale, which the standard bootstrapping
    /// reaches one bit wider for the trace to spend.
    #[allow(clippy::too_many_arguments)]
    fn ckks_replicated_ci_bootstrap_reference<Dst, Src, F, K, E, T>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSReplicatedBootstrapOps<BE>
            + CKKSCopyOps<BE>
            + CKKSRoundOps<BE>
            + CKKSCopyOps<BE, ConjugateInvariant>
            + CKKSRoundOps<BE, ConjugateInvariant>
            + GLWECIEmbed<BE>
            + GLWECITrace<BE>
            + GLWEKeyswitch<BE>,
        Dst: GLWEToBackendMut<BE, ConjugateInvariant> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, ConjugateInvariant> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend> CKKSReplicatedCIBootstrapReference<BE> for Module<BE> {
    fn ckks_replicated_ci_bootstrap_tmp_bytes_reference<C, I, F, A, T, S, K>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> usize
    where
        Self: CKKSReplicatedBootstrapOps<BE>
            + CKKSCIRingMapOps<BE>
            + CKKSCopyOps<BE>
            + CKKSRoundOps<BE>
            + CKKSCopyOps<BE, ConjugateInvariant>
            + CKKSRoundOps<BE, ConjugateInvariant>,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos,
    {
        let module = self;
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
        let wide_input = ckks(n_ci, k_in + lambda_in, ct_in.meta());
        let traced = ckks(n_ci, k_out + lambda_out - 1, level_meta);
        let ops = module
            .ckks_ci_embed_tmp_bytes(&embedded, conversion_infos)
            .max(CKKSRoundOps::<BE>::ckks_round_tmp_bytes(module, &input, &embedded))
            .max(module.ckks_replicated_bootstrap_tmp_bytes(&output, &input, ctx, atk_infos, tsk_infos, encapsulation_infos))
            .max(CKKSCopyOps::<BE>::ckks_copy_tmp_bytes(module, &wide_output, &output))
            .max(module.ckks_ci_trace_tmp_bytes(&traced, &wide_output, conversion_infos))
            .max(CKKSCopyOps::<BE, ConjugateInvariant>::ckks_copy_tmp_bytes(
                module,
                &wide_input,
                ct_in,
            ))
            .max(CKKSRoundOps::<BE, ConjugateInvariant>::ckks_round_tmp_bytes(
                module, ct_out, &traced,
            ));
        // The embedding holds the conjugate-invariant input beside the standard one, and the
        // trace the traced output beside the standard ones.
        let front =
            ct_bytes::<BE>(&input.glwe_layout) + ct_bytes::<BE>(&wide_input.glwe_layout) + ct_bytes::<BE>(&embedded.glwe_layout);
        let back = [&input, &output, &wide_output, &wide_output]
            .iter()
            .map(|layout| ct_bytes::<BE>(&layout.glwe_layout))
            .sum::<usize>()
            + ct_bytes::<BE>(&traced.glwe_layout);
        front.max(back) + ops
    }

    fn ckks_replicated_ci_bootstrap_reference<Dst, Src, F, K, E, T>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSReplicatedBootstrapOps<BE>
            + CKKSCopyOps<BE>
            + CKKSRoundOps<BE>
            + CKKSCopyOps<BE, ConjugateInvariant>
            + CKKSRoundOps<BE, ConjugateInvariant>
            + GLWECIEmbed<BE>
            + GLWECITrace<BE>
            + GLWEKeyswitch<BE>,
        Dst: GLWEToBackendMut<BE, ConjugateInvariant> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, ConjugateInvariant> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        let module = self;
        let (embed_key, trace_key) = conversion_keys;
        const OP: &str = "ckks_replicated_ci_bootstrap";
        check_destination(OP, dst, src)?;
        let plan = ctx.plan;
        if src.slot_layout() != SlotLayout::Replicated(plan.layout) {
            return Err(invalid(OP, "the source must have the bootstrapping's replicated layout"));
        }
        if module.n() != 2 * src.n().as_usize()
            || embed_key.n().as_usize() != module.n()
            || trace_key.n().as_usize() != module.n()
        {
            return Err(invalid(
                OP,
                "takes a module and conversion keys of degree 2N and ciphertexts of degree N",
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
        let (mut input, mut scratch) = scratch
            .borrow()
            .take_ckks_ciphertext_scratch::<_, Standard>(&standard(src.k().as_usize()), src.meta());

        // 1. Into the standard ring, `lambda_in` bits under the scale.
        scratch.scope(|scratch| -> Result<()> {
            let (mut wide, mut scratch) = scratch
                .take_ckks_ciphertext_scratch::<_, ConjugateInvariant>(&invariant(src.k().as_usize() + lambda_in), src.meta());
            CKKSCopyOps::<BE, ConjugateInvariant>::ckks_copy(module, &mut wide, src, &mut scratch)?;
            wide.set_log_delta(src.log_delta() + lambda_in);
            let meta = GLWEPlaintextMeta {
                slots: SlotsKind::Real,
                ..wide.meta()
            };
            let (mut embedded, mut scratch) =
                scratch.take_ckks_ciphertext_scratch::<_, Standard>(&standard(wide.k().as_usize()), meta);
            // `ckks_ci_embed`, the conjugate-invariant operand carved from the same arena.
            module.glwe_ci_embed(&mut embedded, &wide);
            module.glwe_keyswitch_assign(&mut embedded, &embed_key.to_backend_ref(), &mut scratch);
            input.set_k((embedded.log_budget() + src.log_delta()).into());
            CKKSRoundOps::<BE>::ckks_round(module, &mut input, &embedded, &mut scratch)
        })?;

        // 2. The standard bootstrapping, one bit wider than `dst`.
        let level_meta = GLWEPlaintextMeta {
            scale: Scale::Log(plan.level_log_delta),
            ..src.meta()
        };
        let k_out = dst.k().as_usize() + 1;
        let (mut output, mut scratch) = scratch.take_ckks_ciphertext_scratch::<_, Standard>(&standard(k_out), level_meta);
        module.ckks_replicated_bootstrap(&mut output, &input, ctx, workspace, keys, &mut scratch)?;

        // 3. Back by the relative trace, `lambda_out` bits under the scale.
        let (mut wide, mut scratch) =
            scratch.take_ckks_ciphertext_scratch::<_, Standard>(&standard(k_out + lambda_out), output.meta());
        CKKSCopyOps::<BE>::ckks_copy(module, &mut wide, &output, &mut scratch)?;
        wide.set_log_delta(output.log_delta() + lambda_out);
        // `ckks_ci_trace` into a conjugate-invariant operand carved from the same arena:
        // the trace doubles the real part, which one bit less of budget halves.
        let (mut switched, mut scratch) = scratch.take_ckks_ciphertext_scratch::<_, Standard>(&wide.glwe_layout(), wide.meta());
        module.glwe_keyswitch(&mut switched, &wide, &trace_key.to_backend_ref(), &mut scratch);
        let meta = GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..switched.meta()
        };
        let (mut traced, mut scratch) =
            scratch.take_ckks_ciphertext_scratch::<_, ConjugateInvariant>(&invariant(k_out + lambda_out - 1), meta);
        traced.set_log_budget(switched.log_budget() - 1);
        module.glwe_ci_trace(&mut traced, &switched, &mut scratch);
        dst.set_k((traced.log_budget() + plan.level_log_delta).into());
        CKKSRoundOps::<BE, ConjugateInvariant>::ckks_round(module, dst, &traced, &mut scratch)
    }
}
