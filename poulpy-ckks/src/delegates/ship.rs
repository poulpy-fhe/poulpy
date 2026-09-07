use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEKeyswitch, GLWEZero,
    default::keyswitching::glwe::GGLWEProductDefault,
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApplyTmpA,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSShipOps, CKKSSubOps, ShipScalar},
    default::ship::{
        bootstrap::{ship_bootstrap_complex_into, ship_bootstrap_into},
        preflight::ship_bootstrap_tmp_bytes,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
};

impl<BE, F> CKKSShipOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + GGLWEProductDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
{
    fn ckks_ship_bootstrap_tmp_bytes<Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    ) -> Result<usize>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ship_bootstrap_tmp_bytes::<BE, F, _>(self, output, input, keys)
    }

    fn ckks_ship_coeff_encodings_tmp_bytes(&self, plan: &ShipPlan, base2k: Base2K, complex: bool) -> Result<usize> {
        BE::ckks_ship_coeff_encodings_tmp_bytes_impl::<F>(self, plan, base2k, complex)
    }

    fn ckks_ship_coeff_encodings<Src>(
        &self,
        ciphertext: &Src,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        let required = BE::ckks_ship_coeff_encodings_tmp_bytes_impl::<F>(self, plan, base2k, complex)?;
        ckks_ensure!(
            scratch.available() >= required,
            "SHIP coefficient encoding needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );
        BE::ckks_ship_coeff_encodings_impl::<F, Src>(self, ciphertext, plan, base2k, complex, scratch)
    }

    fn ckks_ship_bootstrap_into<Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ship_bootstrap_into::<BE, F, _>(self, output, input, keys, scratch)
    }

    fn ckks_ship_bootstrap_complex_into<Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ship_bootstrap_complex_into::<BE, F, _>(self, output, input, keys, scratch)
    }
}
