use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSLayout,
    api::CKKSFoldOps,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeys},
    oep::CKKSFoldImpl,
    reference::fold::CKKSFoldRing,
};

impl<BE: Backend + CKKSFoldImpl> CKKSFoldOps<BE> for Module<BE> {
    fn ckks_fold_count<IN>(&self, input_module: &Module<IN>, ins: &[CKKSCiphertextOwned<IN>]) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
    {
        BE::ckks_fold_count_impl(self, input_module, ins)
    }

    fn ckks_fold_layout<IN, C, K>(&self, input_module: &Module<IN>, ct_in: &C, keys: &K) -> CKKSLayout
    where
        IN: Backend,
        C: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>,
    {
        BE::ckks_fold_layout_impl(self, input_module, ct_in, keys)
    }

    fn ckks_fold_tmp_bytes<IN, C1, C2, K>(&self, input_module: &Module<IN>, ct_out: &C1, ct_in: &C2, keys: &K) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>,
    {
        BE::ckks_fold_tmp_bytes_impl(self, input_module, ct_out, ct_in, keys)
    }

    fn ckks_fold<IN, K>(
        &self,
        input_module: &Module<IN>,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        K: CKKSFoldKeys<BE, IN>,
    {
        BE::ckks_fold_impl(self, input_module, folded, ins, keys, scratch)
    }

    fn ckks_unfold<IN, K>(
        &self,
        input_module: &Module<IN>,
        outs: &mut [CKKSCiphertextOwned<IN>],
        refreshed: &[CKKSCiphertextOwned<BE>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        K: CKKSFoldKeys<BE, IN>,
    {
        BE::ckks_unfold_impl(self, input_module, outs, refreshed, ins, keys, scratch)
    }
}
