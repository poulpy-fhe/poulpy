//! Encodings validate their operands, then compose the selected slot transforms and coefficient codecs.
use poulpy_core::layouts::{GLWEInfos, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSEncodingScalar, CKKSModuleInfos},
    ckks_ensure_operand,
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferBackendMut, CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef,
        CKKSEncodingBufferViewMut, slot_coeff_count,
    },
    oep::CKKSEncodingImpl,
};

/// Plan-cache key of the encoding plans at precision `F`.
struct CKKSEncodingPlanKey<F>(std::marker::PhantomData<fn() -> F>);

fn validate_coefficients<P: GLWEInfos>(op: &'static str, max_n: usize, pt: &P, coeff_count: usize) -> Result<()> {
    let n = pt.n().as_usize();
    ckks_ensure_operand!(op, pt.rank().as_usize() == 0, "CKKS plaintext encoding expects rank zero");
    ckks_ensure_operand!(op, n <= max_n, "plaintext degree {n} exceeds module capacity {max_n}");
    ckks_ensure_operand!(
        op,
        coeff_count == n,
        "coefficient count {coeff_count} must equal plaintext degree {n}"
    );
    Ok(())
}

fn validate_transform(op: &'static str, max_n: usize, len: usize) -> Result<()> {
    ckks_ensure_operand!(
        op,
        len >= 2 && len.is_power_of_two(),
        "encoding buffer length must be a power of two >= 2, got {len}"
    );
    ckks_ensure_operand!(
        op,
        len <= max_n,
        "encoding buffer length {len} exceeds module capacity {max_n}"
    );
    Ok(())
}

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

pub(crate) fn ckks_encode_slots_assign_into_derived<BE, F, P, C>(module: &Module<BE>, pt: &mut P, slots: &mut C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    validate_transform("ckks_encode_slots_assign_into", 2 * module.ckks_max_slots(), slots.len())?;
    let count = slot_coeff_count(module, slots.len());
    validate_coefficients("ckks_encode_slots_assign_into", module.max_n(), pt, count)?;
    BE::ckks_slots_to_coeffs_assign(module, slots)?;
    BE::ckks_encode_coeffs_into(module, pt, &prefix(&mut slots.to_backend_mut(), count))
}

pub(crate) fn ckks_decode_slots_into_derived<BE, F, P, C>(module: &Module<BE>, pt: &P, slots: &mut C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    validate_transform("ckks_decode_slots_into", 2 * module.ckks_max_slots(), slots.len())?;
    let count = slot_coeff_count(module, slots.len());
    validate_coefficients("ckks_decode_slots_into", module.max_n(), pt, count)?;
    BE::ckks_decode_coeffs_into(module, pt, &mut prefix(&mut slots.to_backend_mut(), count))?;
    BE::ckks_coeffs_to_slots_assign(module, slots)
}

pub(crate) fn ckks_encode_coeffs_into_derived<BE, F, P, C>(module: &Module<BE>, pt: &mut P, coeffs: &C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos + SetCKKSInfos,
    C: CKKSEncodingBufferToBackendRef<BE, F>,
{
    validate_coefficients("ckks_encode_coeffs_into", module.max_n(), pt, coeffs.len())?;
    BE::ckks_quantize_coeffs_into(module, pt, &coeffs.to_backend_ref())
}

pub(crate) fn ckks_decode_coeffs_into_derived<BE, F, P, C>(module: &Module<BE>, pt: &P, coeffs: &mut C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    validate_coefficients("ckks_decode_coeffs_into", module.max_n(), pt, coeffs.len())?;
    BE::ckks_dequantize_coeffs_into(module, pt, &mut coeffs.to_backend_mut())
}

pub(crate) fn ckks_slots_to_coeffs_assign_derived<BE, F, C>(module: &Module<BE>, values: &mut C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    validate_transform("ckks_slots_to_coeffs_assign", 2 * module.ckks_max_slots(), values.len())?;
    let mut values = values.to_backend_mut();
    BE::ckks_encoding_plan_cache(module).with_or_create::<CKKSEncodingPlanKey<F>, _, _>(
        || BE::ckks_encoding_plans_create(module).map_err(::anyhow::Error::from),
        |plans| BE::ckks_slots_to_coeffs_assign_with_plan(module, plans, &mut values),
    )?
}

pub(crate) fn ckks_coeffs_to_slots_assign_derived<BE, F, C>(module: &Module<BE>, values: &mut C) -> Result<()>
where
    BE: CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    C: CKKSEncodingBufferToBackendMut<BE, F>,
{
    validate_transform("ckks_coeffs_to_slots_assign", 2 * module.ckks_max_slots(), values.len())?;
    let mut values = values.to_backend_mut();
    BE::ckks_encoding_plan_cache(module).with_or_create::<CKKSEncodingPlanKey<F>, _, _>(
        || BE::ckks_encoding_plans_create(module).map_err(::anyhow::Error::from),
        |plans| BE::ckks_coeffs_to_slots_assign_with_plan(module, plans, &mut values),
    )?
}
