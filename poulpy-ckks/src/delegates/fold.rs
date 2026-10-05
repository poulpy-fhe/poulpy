use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    Degree, GGLWEInfos, GLWELayout, GLWEToBackendMut, GetAutomorphismKey, prepared::GGLWEPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::{CKKSFoldLayoutOps, CKKSFoldOps},
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSRingCiphertext},
    oep::{CKKSFoldImpl, CKKSFoldLayoutImpl},
};

impl<BE: Backend + CKKSFoldLayoutImpl> CKKSFoldLayoutOps<BE> for Module<BE> {
    fn ckks_fold_layout<C>(&self, ct_in: &C, degree: Degree, keys: &CKKSFoldKeysLayout) -> GLWELayout
    where
        C: CKKSCtBounds,
    {
        BE::ckks_fold_layout_impl(self, ct_in, degree, keys)
    }

    fn ckks_fold_tmp_bytes<C1, C2>(&self, ct_out: &C1, ct_in: &C2, degree: Degree, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        BE::ckks_fold_tmp_bytes_impl(self, ct_out, ct_in, degree, keys)
    }
}

impl<BE: Backend + CKKSFoldImpl<R>, R: Ring> CKKSFoldOps<BE, R> for Module<BE> {
    fn ckks_fold_count(&self, ins: &[CKKSRingCiphertext<BE, R>], degree: Degree) -> usize {
        BE::ckks_fold_count_impl(self, ins, degree)
    }

    fn ckks_unfold_galois_elements<C>(&self, ct_in: &C) -> Vec<i64>
    where
        C: CKKSCtBounds,
    {
        BE::ckks_unfold_galois_elements_impl(self, ct_in)
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
        BE::ckks_fold_impl(self, folded, ins, inbound, scratch)?;
        for ct in folded {
            GLWEToBackendMut::<BE>::set_encryption_metadata(ct, None);
        }
        Ok(())
    }

    fn ckks_unfold<S, H>(
        &self,
        outs: &mut [CKKSRingCiphertext<BE, R>],
        folded: &mut [CKKSCiphertextOwned<BE>],
        outbound: Option<&S>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_unfold_impl(self, outs, folded, outbound, automorphisms, scratch)?;
        for ct in outs {
            GLWEToBackendMut::<BE>::set_encryption_metadata(&mut ct.inner, None);
        }
        for ct in folded {
            GLWEToBackendMut::<BE>::set_encryption_metadata(ct, None);
        }
        Ok(())
    }
}
