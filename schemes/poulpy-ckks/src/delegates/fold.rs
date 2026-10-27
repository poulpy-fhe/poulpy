use crate::CKKSResult as Result;
use poulpy_core::layouts::{Degree, GGLWEInfos, GLWELayout, GetAutomorphismKey, prepared::GGLWEPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::{CKKSFoldLayoutOps, CKKSFoldOps},
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSRingCiphertext},
    oep::{CKKSFoldImpl, CKKSFoldLayoutImpl},
};

impl<BE: Backend + CKKSFoldLayoutImpl> CKKSFoldLayoutOps<BE> for Module<BE> {
    fn ckks_fold_layout<C>(&self, degree: Degree, ct_in: &C, keys: &CKKSFoldKeysLayout) -> GLWELayout
    where
        C: CKKSCtBounds,
    {
        BE::ckks_fold_layout(self, degree, ct_in, keys)
    }

    fn ckks_fold_tmp_bytes<C1, C2>(&self, degree: Degree, ct_out: &C1, ct_in: &C2, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_fold_tmp_bytes(self, degree, ct_out, ct_in, keys)
    }
}

impl<BE: Backend + CKKSFoldImpl<R>, R: Ring> CKKSFoldOps<BE, R> for Module<BE> {
    fn ckks_fold_count(&self, degree: Degree, ins: &[CKKSRingCiphertext<BE, R>]) -> usize {
        BE::ckks_fold_count(self, degree, ins)
    }

    fn ckks_unfold_galois_elements<C>(&self, ct_in: &C) -> Vec<i64>
    where
        C: CKKSCtBounds,
    {
        BE::ckks_unfold_galois_elements(self, ct_in)
    }

    fn ckks_fold<S>(
        &self,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSRingCiphertext<BE, R>],
        inbound: Option<&S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        BE::ckks_fold(self, folded, ins, inbound, scratch)
    }

    fn ckks_unfold<S, H>(
        &self,
        outs: &mut [CKKSRingCiphertext<BE, R>],
        folded: &mut [CKKSCiphertextOwned<BE>],
        automorphisms: Option<&H>,
        outbound: Option<&S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_unfold(self, outs, folded, automorphisms, outbound, scratch)
    }
}
