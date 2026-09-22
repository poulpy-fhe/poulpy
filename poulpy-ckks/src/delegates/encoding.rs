use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::GLWEInfos;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module};

use crate::api::CKKSModuleInfos;
use crate::layouts::CKKSEncodingBuffer;
use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef,
    api::{CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef},
    oep::CKKSEncodingImpl,
};

struct CKKSEncodingPlanKey<F>(std::marker::PhantomData<fn() -> F>);

fn validate_coefficients<P: GLWEInfos>(max_n: usize, pt: &P, coeff_count: usize) -> Result<()> {
    let n = pt.n().as_usize();
    ckks_ensure!(pt.rank().as_usize() == 0, "CKKS plaintext encoding expects rank zero");
    ckks_ensure!(n <= max_n, "plaintext degree {n} exceeds module capacity {max_n}");
    ckks_ensure!(coeff_count > 0, "coefficient count must be non-zero");
    ckks_ensure!(
        coeff_count <= n && n.is_multiple_of(coeff_count),
        "coefficient count {coeff_count} must divide plaintext degree {n}"
    );
    Ok(())
}

fn validate_transform(max_n: usize, len: usize) -> Result<()> {
    ckks_ensure!(
        len >= 2 && len.is_power_of_two(),
        "encoding buffer length must be a power of two >= 2, got {len}"
    );
    ckks_ensure!(len <= max_n, "encoding buffer length {len} exceeds module capacity {max_n}");
    Ok(())
}

impl<BE, F> CKKSEncodingOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    fn ckks_encode_slots_assign_into<P, C>(&self, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        self.ckks_ring().check_plaintext("ckks_encode_slots_assign_into", pt)?;
        validate_transform(2 * self.ckks_max_slots(), slots.len())?;
        let count = if self.ckks_is_conjugate_invariant() {
            slots.len() / 2
        } else {
            slots.len()
        };
        validate_coefficients(self.max_n(), pt, count)?;
        self.ckks_slots_to_coeffs_assign(slots)?;
        let slots = slots.to_backend_ref();
        let coeffs = CKKSEncodingBuffer::from_data(
            BE::region_ref(&slots.data, 0, CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(count)),
            count,
        );
        BE::ckks_encode_coeffs_into_impl(self, pt, &coeffs)
    }

    fn ckks_decode_slots_into<P, C>(&self, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        self.ckks_ring().check_plaintext("ckks_decode_slots_into", pt)?;
        validate_transform(2 * self.ckks_max_slots(), slots.len())?;
        let count = if self.ckks_is_conjugate_invariant() {
            slots.len() / 2
        } else {
            slots.len()
        };
        validate_coefficients(self.max_n(), pt, count)?;
        {
            let mut slots = slots.to_backend_mut();
            let mut coeffs = CKKSEncodingBuffer::from_data(
                BE::region_mut_ref(&mut slots.data, 0, CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(count)),
                count,
            );
            BE::ckks_decode_coeffs_into_impl(self, pt, &mut coeffs)?;
        }
        self.ckks_coeffs_to_slots_assign(slots)
    }

    fn ckks_encode_coeffs_into<P, C>(&self, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendRef<BE, F>,
    {
        self.ckks_ring().check_coefficients("ckks_encode_coeffs_into", pt)?;
        validate_coefficients(self.max_n(), pt, coeffs.len())?;
        BE::ckks_encode_coeffs_into_impl(self, pt, &coeffs.to_backend_ref())
    }

    fn ckks_decode_coeffs_into<P, C>(&self, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<BE> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        self.ckks_ring().check_coefficients("ckks_decode_coeffs_into", pt)?;
        validate_coefficients(self.max_n(), pt, coeffs.len())?;
        BE::ckks_decode_coeffs_into_impl(self, pt, &mut coeffs.to_backend_mut())
    }

    fn ckks_slots_to_coeffs_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        validate_transform(2 * self.ckks_max_slots(), values.len())?;
        let mut values = values.to_backend_mut();
        BE::ckks_encoding_plan_cache_impl(self).with_or_create::<CKKSEncodingPlanKey<F>, _, _>(
            || BE::ckks_encoding_plans_create_impl(self).map_err(::anyhow::Error::from),
            |plans| BE::ckks_slots_to_coeffs_assign_impl(self, plans, &mut values),
        )?
    }

    fn ckks_coeffs_to_slots_assign<C>(&self, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<BE, F>,
    {
        validate_transform(2 * self.ckks_max_slots(), values.len())?;
        let mut values = values.to_backend_mut();
        BE::ckks_encoding_plan_cache_impl(self).with_or_create::<CKKSEncodingPlanKey<F>, _, _>(
            || BE::ckks_encoding_plans_create_impl(self).map_err(::anyhow::Error::from),
            |plans| BE::ckks_coeffs_to_slots_assign_impl(self, plans, &mut values),
        )?
    }
}
