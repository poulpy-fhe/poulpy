//! Slot encodings compose the selected slot transforms and coefficient mappings.
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferBackendMut, CKKSEncodingBufferToBackendMut, CKKSEncodingBufferViewMut,
        slot_coeff_count,
    },
};

fn prefix<'a, BE: Backend + 'a, F>(
    buf: &'a mut CKKSEncodingBufferBackendMut<'_, BE, F>,
    len: usize,
) -> CKKSEncodingBufferViewMut<'a, BE, F> {
    let bytes = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(len);
    CKKSEncodingBufferViewMut::from_inner(CKKSEncodingBuffer::from_data(
        BE::region_mut_ref(&mut buf.data, 0, bytes),
        len,
    ))
}

pub(crate) fn ckks_encode_slots_assign_into<BE, F, P, C>(module: &Module<BE>, pt: &mut P, slots: &mut C) -> Result<()>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSEncodingOps<BE, F>,
    P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    let count = slot_coeff_count(module, slots.len());
    module.ckks_slots_to_coeffs_assign(slots)?;
    module.ckks_encode_coeffs_into(pt, &prefix(&mut slots.to_backend_mut(), count))
}

pub(crate) fn ckks_decode_slots_into<BE, F, P, C>(module: &Module<BE>, pt: &P, slots: &mut C) -> Result<()>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSEncodingOps<BE, F>,
    P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    let count = slot_coeff_count(module, slots.len());
    module.ckks_decode_coeffs_into(pt, &mut prefix(&mut slots.to_backend_mut(), count))?;
    module.ckks_coeffs_to_slots_assign(slots)
}
