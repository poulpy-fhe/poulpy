//! Reference S2C-first bootstrapping with EvalMod on the conjugate-invariant ring
//! (see [`crate::layouts::bootstrapping_ci`]).

use crate::{CKKSResult as Result, ckks_ensure, reference::first_factor::glwe_automorphism_keyless};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        GLWE, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        Rank, SetBSGSMeta, SetK,
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
        BootstrappingCIContext, BootstrappingCIKeysLayout, BootstrappingCIKeysPrepared, BootstrappingKeys,
        BootstrappingKeysPrepared, CKKSCiphertextOwned, CKKSModuleAlloc, CoveringKeys,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
    reference::{
        bootstrapping::ckks_bootstrap_s2c_mod_up,
        ring_map::{ckks_ci_trace_doubled, ckks_ci_trace_doubled_tmp_bytes},
    },
};

/// Layout at the degree of `module` of a ciphertext like `ct`.
fn layout_at<BE: Backend, C: CKKSCtBounds>(module: &Module<BE>, ct: &C) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: module.n().into(),
            base2k: ct.base2k(),
            k: ct.k(),
            rank: Rank(1),
        },
        meta: ct.meta(),
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
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BCI>: GLWEToBackendMut<BCI> + GLWEToBackendRef<BCI> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    crate::layouts::CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos,
    crate::layouts::CKKSPlaintextOwned<BCI>:
        GLWEToBackendRef<BCI> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + poulpy_core::layouts::BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BCI>: GLWETensorKeyPreparedToBackendRef<BCI> + poulpy_core::layouts::GGLWEInfos,
    BootstrappingKeysPrepared<BE::OwnedBuf, BE>: BootstrappingKeys<BE>,
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
        let base = self.ckks_bootstrap_tmp_bytes(ct_out, ct_in, ctx.base(), &keys_layout.base);
        let embedded = layout_at(wide, ct_out);
        let traced = layout_at(ci, ct_out);
        let conversions = wide
            .ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, ctx.pre())
            .max(wide.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, ctx.post()))
            .max(ckks_ci_trace_doubled_tmp_bytes(
                wide,
                &traced,
                &embedded,
                &keys_layout.trace_key,
            ))
            .max(wide.ckks_ci_embed_tmp_bytes(&embedded, &keys_layout.embed_key))
            .max(wide.ckks_add_tmp_bytes(embedded.size()));
        let eval_mod = ci.ckks_eval_mod_tmp_bytes(&traced, &traced, ctx.eval_mod(), &keys_layout.tensor_key);
        (base.max(conversions), eval_mod)
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
        let base_keys = &keys.base;

        // SlotsToCoeffs and ModUp, then CoeffsToSlots without the split: the
        // complex slots `z_j = c_j + i·c_{j+N/2}`.
        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k: ct_in.base2k(),
            k: ct_out.k(),
            rank: Rank(1),
        };
        let mut z = self.ckks_ciphertext_alloc_from_glwe_infos(&boot_layout);
        ckks_bootstrap_s2c_mod_up(self, &mut z, ct_in, base, base_keys, scratch)?;
        self.ckks_dft_evaluate_assign(&mut z, base.coeffs_to_slots(), base_keys.rotation_keys(), scratch)?;

        // `z(X²)` times `u`, traced onto the CI ring without halving: `2·Re(u·z)` at the
        // scale of `z`, exact, the split EvalMod input, real and imaginary parts in two
        // slots each. EvalMod is compiled for that scale.
        let mut embedded = wide.ckks_ciphertext_alloc(z.base2k(), z.k());
        switch_degree(wide, &mut embedded, &z);
        embedded.set_meta(z.meta());
        drop(z);
        wide.ckks_mul_pt_vec_assign(&mut embedded, ctx.pre(), scratch)?;
        let mut traced = ci.ckks_ciphertext_alloc(embedded.base2k(), embedded.k());
        ckks_ci_trace_doubled(wide, &mut traced, &embedded, &keys.trace_key, scratch)?;

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
        wide.ckks_ci_embed(&mut embedded, &reduced, &keys.embed_key, scratch)?;
        drop(reduced);
        wide.ckks_mul_pt_vec_assign(&mut embedded, ctx.post(), scratch)?;
        let mut negated = wide.ckks_ciphertext_alloc(embedded.base2k(), embedded.k());
        glwe_automorphism_keyless(wide, &mut negated, (2 * wide.n() + 1) as i64, &embedded);
        negated.set_meta(embedded.meta());
        wide.ckks_add_assign(&mut embedded, &negated, scratch)?;

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
        Ok(())
    }
}
