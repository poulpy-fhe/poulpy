use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSBootstrappingOps},
    layouts::{
        BootstrappingCIKeys, BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned,
        CKKSPlaintextOwned, EncodedLut, EvalModPlan,
    },
    oep::{CKKSBootstrappingCIImpl, CKKSBootstrappingImpl},
};

impl<BE: Backend + CKKSBootstrappingImpl> CKKSBootstrappingOps<BE> for Module<BE> {
    fn ckks_mod_up_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_mod_up_tmp_bytes(self, res_size)
    }

    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_functional_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, luts, keys_layout)
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
        BE::ckks_mod_up_into(self, dst, src, eval_mod, scratch)
    }

    fn ckks_bootstrap_mod_up_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_bootstrap_mod_up_tmp_bytes(self, ct_out, ct_in, ctx, keys_layout)
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
        BE::ckks_bootstrap_mod_up(self, dst, src, eval_mod, keys, scratch)
    }

    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
    {
        BE::ckks_bootstrap(self, ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_functional_bootstrap<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_functional_bootstrap(self, ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}

impl<BE> CKKSBootstrappingCIOps<BE> for Module<BE>
where
    BE: Backend + CKKSBootstrappingCIImpl,
{
    type CI = BE::CI;

    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_bootstrap_ci_tmp_bytes(self, ci, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_bootstrap_ci_into<Dst, Src, F, K>(
        &self,
        ci: &Module<BE::CI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BE::CI>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingCIKeys<BE, BE::CI>,
    {
        BE::ckks_bootstrap_ci_into(self, ci, dst, src, ctx, keys, scratch, scratch_ci)
    }
}
