use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingOps, CKKSDFTOps, CKKSEvalModOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSPlaintextOwned, EncodedLut,
    },
    oep::CKKSBootstrappingImpl,
};

impl<BE: Backend + CKKSBootstrappingImpl<BE>> CKKSBootstrappingOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE>,
{
    fn ckks_mod_up_tmp_bytes(&self) -> usize {
        BE::ckks_mod_up_tmp_bytes_impl(self)
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
        lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_functional_bootstrap_tmp_bytes_impl(self, ct_out, ct_in, ctx, lut, keys_layout)
    }

    fn ckks_functional_bootstrap_multi_tmp_bytes<C1, C2, F>(
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
        BE::ckks_functional_bootstrap_multi_tmp_bytes_impl(self, ct_out, ct_in, ctx, luts, keys_layout)
    }

    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mod_up_into_impl(self, dst, src, scratch)
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
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_bootstrap_impl::<F, K>(self, ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_bootstrap_real<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_bootstrap_real_impl::<F, K>(self, ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_functional_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_functional_bootstrap_impl::<F, K>(self, ct_out, ct_in, ctx, lut, keys, scratch)
    }

    fn ckks_functional_bootstrap_real<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_functional_bootstrap_real_impl::<F, K>(self, ct_out, ct_in, ctx, lut, keys, scratch)
    }

    fn ckks_functional_bootstrap_multi<F, K>(
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
        BE::ckks_functional_bootstrap_multi_impl::<F, K>(self, ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}
