//! Reference S2C-first bootstrapping with EvalMod on the conjugate-invariant ring
//! (see [`crate::layouts::bootstrapping_ci`]).

use crate::{CKKSResult as Result, ckks_ensure, reference::first_factor::glwe_automorphism_keyless};
use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GLWE, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, Rank, SetBSGSMeta, SetK,
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
        BootstrappingCIContext, BootstrappingCIKeysLayout, BootstrappingCIKeysPrepared, BootstrappingLinearKeys,
        BootstrappingLinearKeysPrepared, CKKSCiphertextOwned, CKKSModuleAlloc, CoveringKeys, ScratchArenaTakeCKKS,
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
    BootstrappingLinearKeysPrepared<BE::OwnedBuf, BE>: BootstrappingLinearKeys<BE>,
    poulpy_core::layouts::GLWECITraceKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE>,
    poulpy_core::layouts::GLWECIEmbedKeyPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE>,
{
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingCIContext<BE, BCI, F>,
        keys_layout: &BootstrappingCIKeysLayout,
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
        let linear = self.glwe_bytes_of_from_infos(&raised)
            + s2c_first_linear_tmp_bytes(self, &raised, ct_in, ctx.base(), &keys_layout.linear);
        let conversions = wide
            .ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, ctx.pre())
            .max(wide.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, ctx.post()))
            .max(ckks_ci_trace_doubled_tmp_bytes(
                wide,
                &traced,
                &embedded,
                &keys_layout.trace_key,
            ))
            .max(wide.ckks_ci_embed_tmp_bytes(&embedded, &keys_layout.embed_key));
        let sum = wide.glwe_bytes_of_from_infos(&embedded) + wide.ckks_add_tmp_bytes(embedded.size());
        let wide_bytes = wide.glwe_bytes_of_from_infos(&embedded) + linear.max(conversions).max(sum);
        let eval_mod = ci.ckks_eval_mod_tmp_bytes(&traced, &traced, ctx.eval_mod(), &keys_layout.tensor_key);
        (wide_bytes, eval_mod)
    }

    fn ckks_bootstrap_ci<F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingCIContext<BE, BCI, F>,
        keys: &BootstrappingCIKeysPrepared<BE::OwnedBuf, BE, BCI>,
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
        let base = ctx.base();
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
                ckks_bootstrap_s2c_mod_up(self, &mut z, ct_in, base, &keys.linear, &mut scratch)?;
                self.ckks_dft_evaluate_assign(&mut z, base.coeffs_to_slots(), keys.linear.rotation_keys(), &mut scratch)?;
                embedded.set_k(z.k());
                switch_degree(wide, &mut embedded, &z);
                embedded.set_meta(z.meta());
                Result::Ok(())
            })?;

            // `z(X²)` times `u`, traced onto the CI ring without halving: `2·Re(u·z)` at
            // the scale of `z`, exact, the split EvalMod input, real and imaginary parts
            // in two slots each; EvalMod is compiled for that scale. The CI ciphertexts
            // pass between the modules of two backends, which share owned buffers only.
            wide.ckks_mul_pt_vec_assign(&mut embedded, ctx.pre(), &mut scratch)?;
            let mut traced = ci.ckks_ciphertext_alloc(embedded.base2k(), embedded.k());
            ckks_ci_trace_doubled(wide, &mut traced, &embedded, &keys.trace_key, &mut scratch)?;

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
            wide.ckks_ci_embed(&mut embedded, &reduced, &keys.embed_key, &mut scratch)?;
            drop(reduced);
            wide.ckks_mul_pt_vec_assign(&mut embedded, ctx.post(), &mut scratch)?;
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
            ct_out.set_log_delta(ct_out.log_delta() - base.c2s_guard_bits());
            ct_out.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(ct_in.log_delta()),
                slots: ct_in.slots(),
                log_sparsity: ct_in.log_sparsity(),
            });
            Result::Ok(())
        })
    }
}
