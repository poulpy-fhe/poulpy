use crate::CKKSResult as Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::{CKKSShipOps, ShipScalar},
    layouts::{CKKSCiphertextOwned, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
    oep::{CKKSEncodingImpl, CKKSShipImpl},
};

impl<BE, F> CKKSShipOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSShipImpl + CKKSEncodingImpl<F>,
    F: ShipScalar,
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
        BE::ckks_ship_bootstrap_tmp_bytes::<F, Src>(self, output, input, keys)
    }

    fn ckks_ship_coeff_encodings_tmp_bytes(&self, plan: &ShipPlan, base2k: Base2K, complex: bool) -> Result<usize> {
        BE::ckks_ship_coeff_encodings_tmp_bytes::<F>(self, plan, base2k, complex)
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
        BE::ckks_ship_coeff_encodings::<F, Src>(self, ciphertext, plan, base2k, complex, scratch)
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
        BE::ckks_ship_bootstrap_into::<F, Src>(self, output, input, keys, scratch)
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
        BE::ckks_ship_bootstrap_complex_into::<F, Src>(self, output, input, keys, scratch)
    }
}
