use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSBootstrappingOps},
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCiphertextOwned,
        CKKSModRaiseInfos, CKKSPlaintextOwned, EncodedLut,
    },
    oep::{CKKSBootstrappingCIImpl, CKKSBootstrappingImpl},
};

impl<BE: Backend + CKKSBootstrappingImpl> CKKSBootstrappingOps<BE> for Module<BE> {
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

    fn ckks_mod_raise_tmp_bytes<R, A, P>(&self, res: &R, a: &A, plan: &P, keys_layout: &BootstrappingKeysLayout) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>,
    {
        BE::ckks_mod_raise_tmp_bytes(self, res, a, plan, keys_layout)
    }

    fn ckks_mod_raise<R, A, P, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>,
    {
        BE::ckks_mod_raise(self, res, a, plan, keys, scratch)
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
    BE: CKKSBootstrappingCIImpl,
{
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
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_bootstrap_ci_into(self, ci, dst, src, ctx, keys, scratch, scratch_ci)
    }
}
