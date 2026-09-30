use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSBootstrapBatchOps, CKKSBootstrappingOps, CKKSDFTOps, CKKSEvalModOps, CKKSFoldOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSFoldKeys, CKKSModuleAlloc, CKKSPlaintextOwned, EncodedLut, EvalModPlan,
    },
    oep::CKKSBootstrappingImpl,
    reference::fold::CKKSFoldRing,
};

impl<BE: Backend + CKKSBootstrappingImpl> CKKSBootstrappingOps<BE> for Module<BE>
where
    Module<BE>: CKKSDFTOps<BE> + CKKSEvalModOps<BE>,
{
    fn ckks_mod_up_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_mod_up_tmp_bytes_impl(self, res_size)
    }

    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_bootstrap_tmp_bytes_impl(self, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_functional_bootstrap_tmp_bytes_impl(self, ct_out, ct_in, ctx, luts, keys_layout)
    }

    fn ckks_mod_up_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mod_up_into_impl(self, dst, src, eval_mod, scratch)
    }

    fn ckks_bootstrap_mod_up<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_bootstrap_mod_up_impl(self, dst, src, eval_mod, keys, scratch)
    }

    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
    {
        BE::ckks_bootstrap_impl(self, ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_functional_bootstrap<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_functional_bootstrap_impl(self, ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}

/// Derived: folding and bootstrapping dispatch through their own families.
impl<BE: Backend> CKKSBootstrapBatchOps<BE> for Module<BE>
where
    Module<BE>: CKKSFoldOps<BE> + CKKSBootstrappingOps<BE> + CKKSModuleAlloc<BE>,
    CKKSCiphertextOwned<BE>: CKKSCtBounds + SetCKKSInfos,
{
    fn ckks_bootstrap_batch_tmp_bytes<IN, C1, C2, F, K>(
        &self,
        input_module: &Module<IN>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
        fold_keys: &K,
    ) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>,
    {
        let folded = self.ckks_fold_layout(input_module, ct_in, fold_keys);
        let refreshed = CKKSLayout {
            glwe_layout: GLWELayout {
                k: ct_out.k(),
                ..folded.glwe_layout
            },
            meta: folded.meta,
        };
        self.ckks_fold_tmp_bytes(input_module, ct_out, ct_in, fold_keys)
            .max(self.ckks_bootstrap_tmp_bytes(&refreshed, &folded, ctx, keys_layout))
    }

    fn ckks_bootstrap_batch<IN, F, K, FK>(
        &self,
        input_module: &Module<IN>,
        outs: &mut [CKKSCiphertextOwned<IN>],
        ins: &[CKKSCiphertextOwned<IN>],
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        fold_keys: &FK,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        F: Sync,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
        FK: CKKSFoldKeys<BE, IN>,
    {
        crate::ckks_ensure!(
            !ins.is_empty() && outs.len() == ins.len(),
            "a batch needs one output per input, got {} outputs for {} inputs",
            outs.len(),
            ins.len()
        );
        crate::ckks_ensure!(
            ctx.functional_message_modulus().is_none(),
            "batch bootstrapping requires an identity recipe"
        );
        crate::ckks_ensure!(
            ctx.pipeline() != BootstrappingPipeline::C2SFirst || ins[0].log_delta() <= ctx.eval_mod().plan.f_mod_log_delta,
            "bootstrap input scale exceeds the C2S-first working scale"
        );
        let layout = self.ckks_fold_layout(input_module, &ins[0], fold_keys);
        let mut folded: Vec<_> = (0..self.ckks_fold_count(input_module, ins))
            .map(|_| self.ckks_ciphertext_alloc_from_glwe_infos(&layout))
            .collect();
        self.ckks_fold(input_module, &mut folded, ins, fold_keys, scratch)?;
        let refreshed_layout = GLWELayout {
            k: outs[0].k(),
            ..layout.glwe_layout
        };
        let mut refreshed = Vec::with_capacity(folded.len());
        for ct in &folded {
            // The folded ciphertext fills the slots its sparsity leaves.
            let log_slots = (self.n().ilog2() as usize - 1).saturating_sub(ct.log_sparsity());
            crate::ckks_ensure!(
                ctx.coeffs_to_slots().plan().log_slots() == log_slots
                    && ctx.slots_to_coeffs().plan().log_slots() == log_slots
                    && ctx
                        .coeffs_to_slots_bypass()
                        .is_none_or(|dft| dft.plan().log_slots() == log_slots),
                "the context transforms must cover the {log_slots}-bit slot count of the folded ciphertexts"
            );
            let mut out = self.ckks_ciphertext_alloc_from_glwe_infos(&refreshed_layout);
            out.set_meta(ct.meta());
            self.ckks_bootstrap(&mut out, ct, ctx, keys, scratch)?;
            refreshed.push(out);
        }
        self.ckks_unfold(input_module, outs, &refreshed, ins, fold_keys, scratch)
    }
}
