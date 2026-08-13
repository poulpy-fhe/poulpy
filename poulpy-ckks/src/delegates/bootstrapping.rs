use crate::CKKSResult as Result;
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
    default::bootstrapping::BootstrappingDefault,
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSModuleAlloc,
        CKKSPlaintextOwned, EncodedLut,
    },
    oep::CKKSEncapsulatedModUpImpl,
};

impl<BE: Backend + CKKSEncapsulatedModUpImpl<BE>> CKKSBootstrappingOps<BE> for Module<BE>
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
        + CKKSPolynomialEvaluationOps<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn ckks_mod_up_tmp_bytes(&self) -> usize {
        BootstrappingDefault::new(self).ckks_mod_up_tmp_bytes_default()
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
        BootstrappingDefault::new(self).ckks_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, keys_layout)
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
        BootstrappingDefault::new(self).ckks_functional_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, luts, keys_layout)
    }

    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BootstrappingDefault::new(self).ckks_mod_up_into_default(dst, src, scratch)
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
        BootstrappingDefault::new(self).ckks_bootstrap_default(ct_out, ct_in, ctx, keys, scratch)
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
        BootstrappingDefault::new(self).ckks_functional_bootstrap_default(ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}
