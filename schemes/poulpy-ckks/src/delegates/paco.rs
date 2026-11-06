use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
};

use crate::{
    CKKSCtBounds,
    api::{CKKSPaCoOps, CKKSPaCoScalar},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, PaCoContext, PaCoKeys, PaCoWorker},
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl, CKKSPaCoImpl},
};

impl<BE, F> CKKSPaCoOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSPaCoImpl + CKKSPaCoCoeffEncodingImpl + CKKSEncodingImpl<F>,
    F: CKKSPaCoScalar,
{
    fn ckks_paco_bootstrap_direct_tmp_bytes<K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_paco_bootstrap_direct_tmp_bytes(self, output, input, context, keys)
    }

    fn ckks_paco_bootstrap_tmp_bytes<K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_paco_bootstrap_tmp_bytes(self, output, input, context, keys)
    }

    fn ckks_paco_coeff_encodings_tmp_bytes<A: GLWEInfos>(&self, context: &PaCoContext<BE, F>, input: &A) -> Result<usize> {
        BE::ckks_paco_coeff_encodings_tmp_bytes(self, context, input)
    }

    fn ckks_paco_coeff_encodings<Src>(
        &self,
        ciphertext: &Src,
        context: &PaCoContext<BE, F>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<[CKKSPlaintextOwned<BE>; 4]>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_paco_coeff_encodings(self, ciphertext, context, scratch)
    }

    fn ckks_paco_bootstrap_direct_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_paco_bootstrap_direct_into(self, output, input, context, keys, scratch)
    }

    fn ckks_paco_bootstrap_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_paco_bootstrap_into(self, output, input, context, keys, scratch)
    }

    fn ckks_paco_bootstrap_parallel_direct_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    {
        BE::ckks_paco_bootstrap_parallel_direct_into(self, output, input, context, keys, workers, scratch)
    }

    fn ckks_paco_bootstrap_parallel_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    {
        BE::ckks_paco_bootstrap_parallel_into(self, output, input, context, keys, workers, scratch)
    }
}
