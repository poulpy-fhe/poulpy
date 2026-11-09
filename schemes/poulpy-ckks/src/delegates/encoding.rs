use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module, Ring};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, SetCKKSInfos,
    api::{CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef},
    oep::CKKSEncodingImpl,
};

impl<BE, F, Rg: Ring> CKKSEncodingOps<BE, F, Rg> for Module<BE>
where
    BE: Backend + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar,
{
    fn ckks_encode_slots_into<P, C>(&self, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE, Rg> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        BE::ckks_encode_slots_into(self, pt, slots)
    }

    fn ckks_decode_slots_into<P, C>(&self, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE, Rg> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        BE::ckks_decode_slots_into(self, pt, slots)
    }

    fn ckks_encode_coeffs_into<P, C>(&self, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE, Rg> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendRef<BE, F>,
    {
        BE::ckks_encode_coeffs_into(self, pt, coeffs)
    }

    fn ckks_decode_coeffs_into<P, C>(&self, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE, Rg> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        BE::ckks_decode_coeffs_into(self, pt, coeffs)
    }

    fn ckks_encoding_buffer_slots_to_coeffs_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        BE::ckks_encoding_buffer_slots_to_coeffs_assign(self, values)
    }

    fn ckks_encoding_buffer_coeffs_to_slots_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        BE::ckks_encoding_buffer_coeffs_to_slots_assign(self, values)
    }
}
