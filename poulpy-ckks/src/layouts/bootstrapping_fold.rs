//! Fold strategies of the batched bootstrapping pipeline.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
        prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena, ZnxWord},
};

use crate::{
    CKKSCtBounds, CKKSError, CKKSInfos, CKKSLayout, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCIRingMapOps, CKKSImagOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSModuleAlloc, RingSwitchKeys,
        validation::{validate_gadget_backend_view, validate_storage_capacity},
    },
    oep::CIBridge,
};

/// Folds a batch of input ciphertexts into standard ciphertexts of the
/// bootstrapping module, refreshes them, and unfolds the results.
///
/// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap)
/// refreshes `ins` into `outs` through a fold, which decides how the inputs
/// are merged into the standard ciphertexts each bootstrap refreshes.
pub trait CKKSBootstrapFold<BE: Backend> {
    /// Input and output ciphertexts.
    type Ciphertext: CKKSCtBounds;

    /// Layouts `(refreshed, folded)` of the standard ciphertexts bootstrapped for
    /// outputs like `ct_out` and inputs like `ct_in`.
    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bound of [`Self::refresh`] for outputs like `ct_out` and inputs like
    /// `ct_in` under keys shaped like `keys_layout`, given the scratch bound
    /// `bootstrap_bytes` of one bootstrap.
    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Folds `ins` into standard ciphertexts, refreshes each with `bootstrap`, and
    /// unfolds the results into `outs`, which has the length of `ins`. `keys` are
    /// the bootstrap keys, available to the fold's own steps.
    #[allow(clippy::too_many_arguments)]
    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>;
}

/// Refreshes standard ciphertexts one per bootstrap.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardFold;

impl<BE: Backend> CKKSBootstrapFold<BE> for StandardFold
where
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
{
    type Ciphertext = CKKSCiphertextOwned<BE>;

    fn bootstrap_layouts<C1, C2>(&self, _module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        (
            CKKSLayout {
                glwe_layout: ct_out.glwe_layout(),
                meta: ct_out.meta(),
            },
            CKKSLayout {
                glwe_layout: ct_in.glwe_layout(),
                meta: ct_in.meta(),
            },
        )
    }

    fn tmp_bytes<C1, C2>(
        &self,
        _module: &Module<BE>,
        _ct_out: &C1,
        _ct_in: &C2,
        _keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        bootstrap_bytes
    }

    fn refresh<F, K, B>(
        &self,
        _module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        _ctx: &BootstrappingContext<BE, F>,
        _keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        for (out, input) in outs.iter_mut().zip(ins) {
            bootstrap(out, input, scratch)?;
        }
        Ok(())
    }
}

/// Refreshes conjugate-invariant ciphertexts of degree `N` through the standard
/// bootstrap of degree `2N`.
///
/// Inputs are taken in pairs, which must share their layout, scale and sparsity:
/// both are unfolded, packed as `left + i·right` and switched once to the standard
/// secret. The refreshed result is switched back to the unfolded CI secret and
/// folded, the real part into the left output and, after a division by `i`, the
/// imaginary part into the right one. An odd tail is refreshed alone. Outputs keep
/// the input scale and sparsity, with real slots.
///
/// The context must use full-slot transforms (`log_slots = log2(N)`) and an
/// identity recipe. Outputs are allocated at `plan.bootstrap_k(output_k + 1, log_delta)`:
/// the extra bit absorbs the factor two of the fold.
pub struct CIFold<'a, CI: Backend, S> {
    module: &'a Module<CI>,
    keys: &'a RingSwitchKeys<S>,
}

impl<'a, CI: Backend, S> CIFold<'a, CI, S> {
    /// Folds through the CI module `module` and the ring-switch `keys`.
    pub fn new(module: &'a Module<CI>, keys: &'a RingSwitchKeys<S>) -> Self {
        Self { module, keys }
    }
}

impl<BE, S> CKKSBootstrapFold<BE> for CIFold<'_, BE::CI, S>
where
    BE: CIBridge,
    Module<BE>: ModuleN + GLWEKeyswitch<BE> + CKKSAddOps<BE> + CKKSImagOps<BE> + CKKSModuleAlloc<BE>,
    Module<BE::CI>: ModuleN + CKKSCIRingMapOps<BE::CI>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE::CI> + GLWEToBackendRef<BE::CI>,
    CKKSCiphertextOwned<BE::CI>: GLWEToBackendMut<BE::CI> + GLWEToBackendRef<BE::CI> + CKKSCtBounds + SetCKKSInfos,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type Ciphertext = CKKSCiphertextOwned<BE::CI>;

    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let layout = |k, meta| CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: self.keys.inbound.base2k(),
                k,
                rank: Rank(1),
            },
            meta,
        };
        (layout(ct_out.k(), ct_out.meta()), layout(ct_in.k(), ct_in.meta()))
    }

    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        _keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let (refreshed, switched) = self.bootstrap_layouts(module, ct_out, ct_in);
        let unfolded = GLWELayout {
            base2k: ct_in.base2k(),
            ..switched.glwe_layout
        };
        let max_size = unfolded.size().max(switched.size()).max(refreshed.size());
        module
            .glwe_keyswitch_tmp_bytes(&switched, &unfolded, &self.keys.inbound)
            .max(module.ckks_add_tmp_bytes(max_size))
            .max(module.ckks_mul_i_tmp_bytes(max_size))
            .max(module.ckks_div_i_tmp_bytes(max_size))
            .max(bootstrap_bytes)
            .max(module.glwe_keyswitch_tmp_bytes(&refreshed, &refreshed, &self.keys.outbound))
            .max(self.module.ckks_ci_fold_tmp_bytes(ct_out, &refreshed))
    }

    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        _keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        let (Some(first_in), Some(first_out)) = (ins.first(), outs.first()) else {
            return Ok(());
        };
        let (ci, keys) = (self.module, self.keys);
        validate_ci_fold(ci, module, outs, ins)?;
        let log_slots = ci.n().ilog2() as usize;
        crate::ckks_ensure!(
            ctx.coeffs_to_slots().plan().log_slots() == log_slots
                && ctx.slots_to_coeffs().plan().log_slots() == log_slots
                && ctx
                    .coeffs_to_slots_bypass()
                    .is_none_or(|dft| dft.plan().log_slots() == log_slots),
            "CI bootstrapping requires full-slot standard transforms"
        );
        crate::ckks_ensure!(
            ctx.functional_message_modulus().is_none(),
            "CI bootstrapping requires an identity recipe"
        );
        crate::ckks_ensure!(
            ctx.pipeline() != BootstrappingPipeline::C2SFirst || first_in.log_delta() <= ctx.eval_mod().plan.f_mod_log_delta,
            "CI bootstrap input scale exceeds the C2S-first working scale"
        );

        let base2k = keys.inbound.base2k();
        crate::ckks_ensure!(
            (1..=<BE::ZnxWord as ZnxWord>::BITS - 2).contains(&base2k.as_usize()),
            "invalid CI switching-key radix"
        );
        crate::ckks_ensure!(base2k == keys.outbound.base2k(), "CI bootstrap switching-key radices differ");
        validate_gadget_backend_view(
            "inbound ring-switch key",
            &keys.inbound,
            &keys.inbound.to_backend_ref(),
            module.n(),
            base2k,
            first_in.k().as_usize().div_ceil(base2k.as_usize()),
        )?;
        crate::ckks_ensure!(
            keys.inbound.gglwe_layout().gadget_k() >= first_in.k(),
            "inbound ring-switch key does not cover the input width"
        );
        // The return switch runs after the standard pipeline has consumed its budget.
        let return_k = first_out
            .k()
            .as_usize()
            .checked_sub(ctx.output_consumed_bits(first_in.log_delta()))
            .ok_or_else(|| CKKSError::from(anyhow::anyhow!("insufficient CI bootstrap output width")))?;
        crate::ckks_ensure!(
            return_k > first_in.log_delta() + 1,
            "CI bootstrap output has no message budget"
        );
        validate_gadget_backend_view(
            "outbound ring-switch key",
            &keys.outbound,
            &keys.outbound.to_backend_ref(),
            module.n(),
            base2k,
            return_k.div_ceil(base2k.as_usize()),
        )?;
        crate::ckks_ensure!(
            keys.outbound.gglwe_layout().gadget_k().as_usize() >= return_k,
            "outbound ring-switch key does not cover the return width"
        );

        let layout = |base2k, k| GLWELayout {
            n: module.n().into(),
            base2k,
            k,
            rank: Rank(1),
        };
        for (outs, ins) in outs.chunks_mut(2).zip(ins.chunks(2)) {
            let input = &ins[0];
            // Unfolded inputs share the unfolded secret, so a pair is packed before one inbound switch.
            let unfolded = layout(input.base2k(), input.k());
            let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(&unfolded);
            ci.ckks_ci_unfold(&mut packed, input)?;
            if let Some(right) = ins.get(1) {
                let mut right_packed = module.ckks_ciphertext_alloc_from_glwe_infos(&unfolded);
                ci.ckks_ci_unfold(&mut right_packed, right)?;
                module.ckks_mul_i_assign(&mut right_packed, scratch)?;
                module.ckks_add_assign(&mut packed, &right_packed, scratch)?;
                packed.set_slots(SlotsKind::Complex);
            }
            let mut switched = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(base2k, input.k()));
            switched.set_meta(packed.meta());
            module.glwe_keyswitch(&mut switched, &packed, &keys.inbound.to_backend_ref(), scratch);

            let mut refreshed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(base2k, outs[0].k()));
            refreshed.set_meta(input.meta());
            bootstrap(&mut refreshed, &switched, scratch)?;
            module.glwe_keyswitch_assign(&mut refreshed, &keys.outbound.to_backend_ref(), scratch);
            // The fold doubles the real part; relabeling at the input scale drops that bit.
            let (left, right) = outs.split_first_mut().expect("chunks are nonempty");
            ci.ckks_ci_fold(left, &refreshed, &mut scratch.borrow().into_backend())?;
            left.set_log_delta(input.log_delta());
            if let Some(right) = right.first_mut() {
                module.ckks_div_i_assign(&mut refreshed, scratch)?;
                ci.ckks_ci_fold(right, &refreshed, &mut scratch.borrow().into_backend())?;
                right.set_log_delta(input.log_delta());
            }
        }
        Ok(())
    }
}

fn validate_ci_fold<CI: Backend, BE: Backend>(
    ci: &Module<CI>,
    module: &Module<BE>,
    outs: &[CKKSCiphertextOwned<CI>],
    ins: &[CKKSCiphertextOwned<CI>],
) -> Result<()>
where
    Module<CI>: ModuleN,
    Module<BE>: ModuleN,
    CKKSCiphertextOwned<CI>: CKKSCtBounds,
{
    crate::ckks_ensure!(
        module.n() == 2 * ci.n(),
        "the standard module degree must be twice the CI module degree"
    );
    let (input, output) = (&ins[0], &outs[0]);
    for ct in ins.iter().chain(outs) {
        validate_storage_capacity("CI bootstrap ciphertext", ct)?;
        crate::ckks_ensure!(
            ct.base2k().as_usize() <= <CI::ZnxWord as ZnxWord>::BITS - 2,
            "CI ciphertext radix exceeds the backend limit"
        );
    }
    crate::ckks_ensure!(
        input.n().as_usize() == ci.n() && output.n().as_usize() == ci.n(),
        "CI ciphertext degree does not match the CI module"
    );
    crate::ckks_ensure!(
        input.rank().as_usize() == 1 && output.rank().as_usize() == 1,
        "CI bootstrapping supports rank-1 ciphertexts only"
    );
    crate::ckks_ensure!(input.log_delta() <= input.k().as_usize(), "CI input scale exceeds its width");
    crate::ckks_ensure!(input.log_sparsity() <= ci.n().ilog2() as usize, "invalid CI input sparsity");
    crate::ckks_ensure!(
        ins.iter().all(|ct| ct.glwe_layout() == input.glwe_layout()
            && ct.log_delta() == input.log_delta()
            && ct.log_sparsity() == input.log_sparsity())
            && outs.iter().all(|ct| ct.glwe_layout() == output.glwe_layout()),
        "CI bootstrap inputs and outputs must have matching layouts, scale and sparsity"
    );
    Ok(())
}
