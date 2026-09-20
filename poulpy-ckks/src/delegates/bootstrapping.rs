use crate::CKKSResult as Result;
use poulpy_core::GLWENormalize;
use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps,
        CKKSEvalModOps, CKKSImagOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::EvalModPlan,
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSModuleAlloc,
        CKKSPlaintextOwned, EncodedLut,
    },
    oep::CKKSEncapsulatedModUpImpl,
    reference::bootstrapping::BootstrappingReference,
};

impl<BE: Backend + CKKSEncapsulatedModUpImpl> CKKSBootstrappingOps<BE> for Module<BE>
where
    Module<BE>: GLWEBytesOf<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSMulOps<BE>
        + CKKSAffineOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + GLWENormalize<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn ckks_mod_up_tmp_bytes(&self, res_size: usize) -> usize {
        BootstrappingReference::new(self).ckks_mod_up_tmp_bytes_reference(res_size)
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
        BootstrappingReference::new(self).ckks_bootstrap_tmp_bytes_reference(ct_out, ct_in, ctx, keys_layout)
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
        BootstrappingReference::new(self).ckks_functional_bootstrap_tmp_bytes_reference(ct_out, ct_in, ctx, luts, keys_layout)
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
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mod_up_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mod_up_into", src)?;
        let scale_up = eval_mod.raised_scale_up(src.log_delta())?;
        BootstrappingReference::new(self).ckks_mod_up_into_reference(dst, src, scale_up, scratch)?;
        // Relabel by the message ratio here, so callers never have to stamp
        // metadata by hand after the call.
        let mut meta = dst.meta();
        meta.log_delta += eval_mod.log_msg_ratio;
        dst.set_meta(meta);
        Ok(())
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
        let ring = crate::api::CKKSModuleInfos::ckks_ring(self);
        check_key_rings::<BE, _>(keys, "ckks_bootstrap_mod_up", ring)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_bootstrap_mod_up", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_bootstrap_mod_up", src)?;
        BootstrappingReference::new(self).ckks_bootstrap_mod_up_reference(dst, src, eval_mod, keys, scratch)
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
        let ring = crate::api::CKKSModuleInfos::ckks_ring(self);
        check_key_rings::<BE, _>(keys, "ckks_bootstrap", ring)?;
        crate::ckks_ensure!(
            ring.kind == crate::layouts::CKKSRingKind::Standard,
            "bootstrapping requires a standard ring"
        );
        for factor in &ctx.coeffs_to_slots().inner().factors {
            ring.check("bootstrap context", factor.ring())?;
        }
        ring.check_ciphertext("ckks_bootstrap", ct_in)?;
        ring.check_ciphertext("ckks_bootstrap", ct_out)?;
        BootstrappingReference::new(self).ckks_bootstrap_reference(ct_out, ct_in, ctx, keys, scratch)
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
        let ring = crate::api::CKKSModuleInfos::ckks_ring(self);
        check_key_rings::<BE, _>(keys, "ckks_functional_bootstrap", ring)?;
        crate::ckks_ensure!(
            ring.kind == crate::layouts::CKKSRingKind::Standard,
            "bootstrapping requires a standard ring"
        );
        for factor in &ctx.coeffs_to_slots().inner().factors {
            ring.check("bootstrap context", factor.ring())?;
        }
        for lut in luts {
            lut.check_ring::<BE>(ring)?;
        }
        ring.check_ciphertext("ckks_functional_bootstrap", ct_in)?;
        for ct in ct_outs.iter() {
            ring.check_ciphertext("ckks_functional_bootstrap", ct)?;
        }
        BootstrappingReference::new(self).ckks_functional_bootstrap_reference(ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}

fn check_key_rings<BE: Backend, K: BootstrappingKeys<BE>>(
    keys: &K,
    op: &'static str,
    ring: crate::layouts::CKKSRing,
) -> crate::CKKSResult<()> {
    ring.check(op, keys.rotation_keys().key_ring())?;
    ring.check(op, keys.tensor_key().key_ring())?;
    if let Some((a, b)) = keys.encapsulation_keys() {
        ring.check(op, a.key_ring())?;
        ring.check(op, b.key_ring())?;
    }
    Ok(())
}
