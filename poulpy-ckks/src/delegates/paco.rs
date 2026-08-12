use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEAutomorphism, GLWEKeyswitch, GLWELinearTransformations, GLWERotate,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena, ScratchOwned},
};

use crate::{
    CKKSCtBounds,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSPaCoOps, CKKSRotateOps,
        CKKSSubOps, PaCoScalar,
    },
    default::paco::{
        ops::PaCoSlotOps,
        parallel::{
            paco_bootstrap_direct_into, paco_bootstrap_into, paco_bootstrap_parallel_direct_into, paco_bootstrap_parallel_into,
        },
        preflight::paco_bootstrap_tmp_bytes,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, PaCoContext, PaCoKeys, PaCoWorker},
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl},
};

impl<BE, F> CKKSPaCoOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSPaCoCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: PaCoScalar,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + PaCoSlotOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWERotate<BE>
        + GLWEAutomorphism<BE>
        + GLWELinearTransformations<BE>
        + GLWEKeyswitch<BE>
        + CyclotomicOrder,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + Send + Sync,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    BE::OwnedBuf: Sync,
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
        paco_bootstrap_tmp_bytes(self, output, input, context, keys, false)
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
        paco_bootstrap_tmp_bytes(self, output, input, context, keys, true)
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
        let required = BE::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(self, context.plan())?;
        ckks_ensure!(
            scratch.available() >= required,
            "PaCo coefficient encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        BE::ckks_paco_coeff_encodings_impl::<F, Src>(self, ciphertext, context.plan(), context.base2k(), scratch)
    }

    fn ckks_paco_coeff_encodings_tmp_bytes(&self, context: &PaCoContext<BE, F>) -> Result<usize> {
        BE::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(self, context.plan())
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
        paco_bootstrap_direct_into::<BE, F, K, Src>(self, output, input, context, keys, scratch)
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
        paco_bootstrap_into::<BE, F, K, Src>(self, output, input, context, keys, scratch)
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
        paco_bootstrap_parallel_direct_into::<BE, F, K, Src>(self, output, input, context, keys, workers, scratch)
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
        paco_bootstrap_parallel_into::<BE, F, K, Src>(self, output, input, context, keys, workers, scratch)
    }
}
