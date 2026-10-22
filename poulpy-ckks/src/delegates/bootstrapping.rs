use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSBootstrappingOps, CKKSDFTOps, CKKSEvalModOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSPlaintextOwned, EncodedLut,
        EvalModPlan,
    },
    oep::{CKKSBootstrappingCIImpl, CKKSBootstrappingImpl},
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
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
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
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_functional_bootstrap_impl(self, ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}

impl<BE, BCI> CKKSBootstrappingCIOps<BE, BCI> for Module<BE>
where
    BE: Backend + CKKSBootstrappingCIImpl,
    BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + for<'a> Backend<BufRef<'a> = BE::BufRef<'a>, BufMut<'a> = BE::BufMut<'a>>
        + 'static,
    Module<BCI>: CKKSEvalModOps<BCI>,
{
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_bootstrap_ci_tmp_bytes_impl(self, wide, ci, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_bootstrap_ci<F, K>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, BCI>,
    {
        BE::ckks_bootstrap_ci_impl(self, wide, ci, ct_out, ct_in, ctx, keys, scratch, scratch_ci)
    }
}
