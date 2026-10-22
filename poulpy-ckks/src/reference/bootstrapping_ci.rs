//! Reference S2C-first bootstrapping with EvalMod on the conjugate-invariant ring
//! (see [`crate::layouts::bootstrapping_ci`]).

use crate::{CKKSCompositionError, CKKSResult as Result, ckks_ensure, reference::first_factor::glwe_automorphism_keyless};
use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWETensorKeyPrepared, GLWEToBackendMut,
        GLWEToBackendRef, GetAutomorphismKey, LWEInfos, Rank, SetBSGSMeta, SetK, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{VecZnxAutomorphism, VecZnxSwitchRing},
    layouts::{Backend, ConjugateInvariant, Module, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSCIRingMapOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps,
        CKKSMulOps, CKKSPow2Ops,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeysLayout, BootstrappingKeysPrepared, BootstrappingLinearKeys, CKKSCiphertextOwned,
        CKKSModuleAlloc, ConversionKeysLayout, CoveringKeys, EvalModRing, ScratchArenaTakeCKKS, bootstrapping_keys::check_covers,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
    reference::{
        bootstrapping::{ckks_bootstrap_s2c_mod_up, s2c_first_linear_tmp_bytes},
        ring_map::{ckks_ci_trace_doubled, ckks_ci_trace_doubled_tmp_bytes},
    },
};

/// Layout of a rank-1 ciphertext of degree `n` like `ct`, at scale `1`: the
/// largest budget, which bounds EvalMod's working width.
fn layout_at<C: CKKSCtBounds>(n: usize, ct: &C) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: n.into(),
            base2k: ct.base2k(),
            k: ct.k(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta::default(),
    }
}

/// Fails before any work unless every key covers the widest ciphertext the CI
/// bootstrap switches with it at the working width `ct_out.k()`, the CI tensor
/// key and the trace and embed keys included.
fn check_ci_key_widths<BE, BCI, F>(
    ct_out: &CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F, BCI>,
    keys: &BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>,
) -> Result<()>
where
    BE: Backend,
    BCI: Backend,
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
    BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>: BootstrappingLinearKeys<BE>,
{
    let op = "ckks_bootstrap_ci";
    let widths = ctx.key_widths();
    let (bootstrap_k, input_k) = (ct_out.k().as_usize(), ct_in.k().as_usize());
    let log_n = ct_out.n().as_usize().trailing_zeros() as usize;
    let rotations = CoveringKeys(keys.rotation_keys());
    for (p, k) in widths.rotation_key_inputs(log_n, bootstrap_k, input_k) {
        let k = TorusPrecision(k as u32);
        rotations
            .get_automorphism_key(p, k)
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation: p,
                k: k.into(),
                source,
            })?;
    }
    let EvalModRing::ConjugateInvariant(conversion) = ctx.eval_mod_ring() else {
        crate::ckks_bail!("{op}: the context runs EvalMod on the ring of the bootstrap");
    };
    let consumed = ctx.coeffs_to_slots().consumed_bits() + conversion.pre_log_delta + ctx.eval_mod().plan.consumed_bits();
    ckks_ensure!(
        bootstrap_k >= consumed,
        "{op}: a working width of {bootstrap_k} bits is below the {consumed} that CoeffsToSlots, the conversion and EvalMod consume"
    );
    let (traced, embedded) = widths
        .conversions(bootstrap_k)
        .expect("EvalMod on the CI ring has conversions");
    let k = TorusPrecision(widths.eval_mod_relinearization_k(traced, ct_in.log_delta()) as u32);
    check_covers(op, keys.tensor_key.gglwe_layout(), k).map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
        op,
        k: k.into(),
        source,
    })?;
    let Some((trace_key, embed_key)) = &keys.conversion_keys else {
        crate::ckks_bail!("{op}: the keys have no conversion keys");
    };
    for (name, layout, k) in [
        ("trace", trace_key.gglwe_layout(), traced),
        ("embed", embed_key.gglwe_layout(), embedded),
    ] {
        check_covers(op, layout, TorusPrecision(k as u32)).map_err(|source| CKKSCompositionError::InvalidOperand {
            op,
            reason: format!("the {name} key does not cover its input: {source}"),
        })?;
    }
    crate::reference::bootstrapping::check_raise_keys(ct_out, ct_in, ctx, keys)
}

/// `res` = `a` at the degree of `res`, column by column: `X -> X^(n_res / n_a)` when
/// it is larger, the coefficients at multiples of the ratio when it is smaller.
fn switch_degree<BE, R, A>(module: &Module<BE>, res: &mut R, a: &A)
where
    BE: Backend,
    Module<BE>: VecZnxSwitchRing<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let a_ref = a.to_backend_ref();
    for col in 0..a.rank().as_usize() + 1 {
        module.vec_znx_switch_ring(res.to_backend_mut().data_mut(), col, a_ref.data(), col);
    }
    res.set_canonical(a_ref.is_canonical());
}

impl<BE, BCI> CKKSBootstrappingCIOps<BE, BCI> for Module<BE>
where
    BE: Backend<Ring = Standard> + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
    Module<BE>: CKKSBootstrappingOps<BE>
        + GLWEBytesOf<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSDFTOps<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSCIRingMapOps<BE>
        + poulpy_core::GLWECITrace<BE>
        + CKKSModuleAlloc<BE>
        + VecZnxSwitchRing<BE>
        + VecZnxAutomorphism<BE>,
    Module<BCI>: CKKSEvalModOps<BCI> + CKKSModuleAlloc<BCI>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    CKKSCiphertextOwned<BCI>: GLWEToBackendMut<BCI> + GLWEToBackendRef<BCI> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    crate::layouts::CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
    crate::layouts::CKKSPlaintextOwned<BCI>:
        GLWEToBackendRef<BCI> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + poulpy_core::layouts::BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BCI>: GLWETensorKeyPreparedToBackendRef<BCI> + poulpy_core::layouts::GGLWEInfos,
    BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>: BootstrappingLinearKeys<BE>,
    poulpy_core::layouts::GLWECITraceKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE>,
    poulpy_core::layouts::GLWECIEmbedKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE>,
{
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F, BCI>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let raised = layout_at(self.n(), ct_out);
        let embedded = layout_at(wide.n(), ct_out);
        let traced = layout_at(ci.n(), ct_out);
        // `embedded` lives through the whole bootstrap; `z` through the linear
        // stages, `negated` through the last sum.
        let linear = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&raised))
            + s2c_first_linear_tmp_bytes(self, &raised, ct_in, ctx, keys_layout);
        // A context or layout without conversions fails at the bootstrap.
        let (Some((pre, post)), Some(ConversionKeysLayout { trace_key, embed_key })) =
            (ctx.conversions(), &keys_layout.conversion)
        else {
            return (0, 0);
        };
        let conversions = wide
            .ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, pre)
            .max(wide.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, post))
            .max(ckks_ci_trace_doubled_tmp_bytes(wide, &traced, &embedded, trace_key))
            .max(wide.ckks_ci_embed_tmp_bytes(&embedded, embed_key));
        // Every temporary is carved at an aligned offset.
        let embedded_bytes = BE::scratch_aligned(wide.glwe_bytes_of_from_infos(&embedded));
        let sum = embedded_bytes + wide.ckks_add_tmp_bytes(embedded.size());
        let wide_bytes = embedded_bytes + linear.max(conversions).max(sum);
        let eval_mod = ci.ckks_eval_mod_tmp_bytes(&traced, &traced, ctx.eval_mod(), &keys_layout.tensor_key);
        (wide_bytes, eval_mod)
    }

    fn ckks_bootstrap_ci<F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F, BCI>,
        keys: &BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()> {
        ckks_ensure!(
            wide.n() == 2 * self.n() && ci.n() == self.n(),
            "ckks_bootstrap_ci takes a standard module of degree 2N and a CI module of degree N"
        );
        ckks_ensure!(
            ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
            "ckks_bootstrap_ci supports rank-1 ciphertexts only"
        );
        check_ci_key_widths(ct_out, ct_in, ctx, keys)?;
        let (Some((pre, post)), Some((trace_key, embed_key))) = (ctx.conversions(), &keys.conversion_keys) else {
            crate::ckks_bail!("ckks_bootstrap_ci: the context or the keys have no conversions");
        };
        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k: ct_in.base2k(),
            k: ct_out.k(),
            rank: Rank(1),
        };
        let wide_layout = GLWELayout {
            n: wide.n().into(),
            ..boot_layout
        };

        scratch.scope(|scratch| {
            let (mut embedded, mut scratch) = scratch.take_ckks_ciphertext_scratch(&wide_layout, ct_in.meta());

            // SlotsToCoeffs and ModUp, then CoeffsToSlots without the split: the
            // complex slots `z_j = c_j + i·c_{j+N/2}`, embedded as `z(X²)`.
            scratch.scope(|scratch| {
                let (mut z, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
                ckks_bootstrap_s2c_mod_up(self, &mut z, ct_in, ctx, keys, &mut scratch)?;
                self.ckks_dft_evaluate_assign(
                    &mut z,
                    ctx.coeffs_to_slots(),
                    &CoveringKeys(keys.rotation_keys()),
                    &mut scratch,
                )?;
                embedded.set_k(z.k());
                switch_degree(wide, &mut embedded, &z);
                embedded.set_meta(z.meta());
                Result::Ok(())
            })?;

            // `z(X²)` times `u`, traced onto the CI ring without halving: `2·Re(u·z)`, the
            // split EvalMod input. The CI ciphertexts are owned: the two backends share
            // owned buffers only.
            wide.ckks_mul_pt_vec_assign(&mut embedded, pre, &mut scratch)?;
            let mut traced = ci.ckks_ciphertext_alloc(embedded.base2k(), embedded.k());
            ckks_ci_trace_doubled(wide, &mut traced, &embedded, trace_key, &mut scratch)?;

            // One EvalMod for all `N` coefficients.
            let mut reduced = ci.ckks_ciphertext_alloc(traced.base2k(), traced.k());
            ci.ckks_eval_mod(
                &mut reduced,
                &traced,
                ctx.eval_mod(),
                &CoveringKeys(&keys.tensor_key),
                scratch_ci,
            )?;
            drop(traced);

            // Back to degree `2N`, times `v`, and `x + x(−X)`: `r_j + i·r_{j+N/2}` on the
            // even coefficients, under `s(X²)`, which `X -> −X` leaves unchanged.
            wide.ckks_ci_embed(&mut embedded, &reduced, embed_key, &mut scratch)?;
            drop(reduced);
            wide.ckks_mul_pt_vec_assign(&mut embedded, post, &mut scratch)?;
            scratch.scope(|scratch| {
                let negated_layout = GLWELayout {
                    k: embedded.k(),
                    ..wide_layout
                };
                let (mut negated, mut scratch) = scratch.take_ckks_ciphertext_scratch(&negated_layout, embedded.meta());
                glwe_automorphism_keyless(wide, &mut negated, (2 * wide.n() + 1) as i64, &embedded);
                negated.set_meta(embedded.meta());
                wide.ckks_add_assign(&mut embedded, &negated, &mut scratch)
            })?;

            ct_out.set_k(embedded.k());
            switch_degree(self, ct_out, &embedded);
            // Keep the C2S guard bits through EvalMod, then relabel at the output width.
            ct_out.set_meta(embedded.meta());
            ct_out.set_log_delta(ct_out.log_delta() - ctx.c2s_guard_bits());
            ct_out.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(ct_in.log_delta()),
                slots: ct_in.slots(),
                log_sparsity: ct_in.log_sparsity(),
            });
            Result::Ok(())
        })
    }
}
