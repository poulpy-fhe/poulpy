//! The CKKS scheme layer: the canonical encoding circuit of
//! [`poulpy_ckks::reference::encoding`] over the oracle FFT, the PaCo and SHIP
//! coefficient encodings of `poulpy-ckks`, and the generic CKKS compositions.

use poulpy_ckks::{
    CKKSError, CKKSResult,
    api::CKKSEncodingScalar,
    reference::encoding::{CKKSSlotEmbedding, EncodingPermutation},
};
use poulpy_hal::api::NegacyclicFFTNew;

use crate::{FFT64CIOracle, FFT64Oracle, NTT4x30CIOracle, NTT4x30Oracle, fft::ComplexFft};

/// Encoding plans at precision `S`: the slot permutation and the transform
/// for every power-of-two slot count up to the module capacity, by `log2(slots)`.
pub struct EncodingPlans<S> {
    plans: Vec<(EncodingPermutation, ComplexFft<S>)>,
}

impl<S: CKKSEncodingScalar> EncodingPlans<S> {
    fn new(max_slots: usize) -> CKKSResult<Self> {
        let plans = (0..=max_slots.ilog2())
            .map(|log_slots| Ok((EncodingPermutation::new(1 << log_slots)?, ComplexFft::new(1 << log_slots))))
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { plans })
    }

    fn for_slots(&self, slots: usize) -> CKKSResult<(&EncodingPermutation, &ComplexFft<S>)> {
        if !slots.is_power_of_two() || slots.ilog2() as usize >= self.plans.len() {
            return Err(CKKSError::Internal(anyhow::anyhow!(
                "slot count {slots} is not a power of two within the module capacity"
            )));
        }
        let (map, fft) = &self.plans[slots.ilog2() as usize];
        Ok((map, fft))
    }
}

/// The encoding extension point, entirely from the CKKS reference.
macro_rules! impl_oracle_ckks_encoding {
    ($be:ty) => {
        unsafe impl<S: CKKSEncodingScalar> ::poulpy_ckks::oep::CKKSEncodingImpl<S> for $be {
            type Plans = EncodingPlans<S>;

            fn ckks_encoding_plan_cache_impl(
                module: &::poulpy_hal::layouts::Module<$be>,
            ) -> &::poulpy_hal::layouts::ModulePlanCache {
                crate::backend::plan_cache(module)
            }

            fn ckks_encoding_plans_create_impl(module: &::poulpy_hal::layouts::Module<$be>) -> CKKSResult<Self::Plans> {
                EncodingPlans::new(::poulpy_ckks::api::CKKSModuleInfos::ckks_max_slots(module))
            }

            fn ckks_encode_coeffs_into_impl<P>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                pt: &mut P,
                coeffs: &::poulpy_ckks::layouts::CKKSEncodingBufferBackendRef<'_, $be, S>,
            ) -> CKKSResult<()>
            where
                P: ::poulpy_ckks::CKKSPlaintextToBackendMut<$be>
                    + ::poulpy_core::layouts::IntPolyInfos
                    + ::poulpy_ckks::SetCKKSInfos,
            {
                Ok(::poulpy_ckks::reference::encoding::encode_coeffs_into_host::<$be, S, P>(pt, coeffs)?)
            }

            fn ckks_decode_coeffs_into_impl<P>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                pt: &P,
                coeffs: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, S>,
            ) -> CKKSResult<()>
            where
                P: ::poulpy_ckks::CKKSPlaintextToBackendRef<$be> + ::poulpy_core::layouts::IntPolyInfos,
            {
                Ok(::poulpy_ckks::reference::encoding::decode_coeffs_into_host::<$be, S, P>(pt, coeffs)?)
            }

            fn ckks_slots_to_coeffs_assign_impl(
                _module: &::poulpy_hal::layouts::Module<$be>,
                plans: &Self::Plans,
                values: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, S>,
            ) -> CKKSResult<()> {
                let (map, fft) = plans.for_slots(::poulpy_ckks::layouts::CKKSEncodingBufferInfos::len(values) / 2)?;
                let ring = <<$be as ::poulpy_hal::layouts::Backend>::Ring as CKKSSlotEmbedding>::slots_to_coeffs_assign;
                Ok(ring(map, fft, values.as_mut_slice())?)
            }

            fn ckks_coeffs_to_slots_assign_impl(
                _module: &::poulpy_hal::layouts::Module<$be>,
                plans: &Self::Plans,
                values: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, S>,
            ) -> CKKSResult<()> {
                let (map, fft) = plans.for_slots(::poulpy_ckks::layouts::CKKSEncodingBufferInfos::len(values) / 2)?;
                let ring = <<$be as ::poulpy_hal::layouts::Backend>::Ring as CKKSSlotEmbedding>::coeffs_to_slots_assign;
                Ok(ring(map, fft, values.as_mut_slice())?)
            }
        }
    };
}

/// The PaCo and SHIP coefficient encodings of `poulpy-ckks`, on an owned copy
/// of the ciphertext.
macro_rules! impl_oracle_ckks_coeff_encodings {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for $be {
            fn ckks_paco_coeff_encodings_tmp_bytes_impl<S>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _plan: &::poulpy_ckks::layouts::PaCoPlan,
            ) -> CKKSResult<usize>
            where
                S: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<S>,
            {
                Ok(0)
            }

            fn ckks_paco_coeff_encodings_impl<S, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct: &Src,
                plan: &::poulpy_ckks::layouts::PaCoPlan,
                base2k: ::poulpy_core::layouts::Base2K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> CKKSResult<[::poulpy_ckks::layouts::CKKSPlaintextOwned<$be>; 4]>
            where
                S: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<S>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                use ::poulpy_ckks::layouts::CKKSModuleAlloc;
                use ::poulpy_core::GLWECopy;
                if module.n() != plan.n() {
                    return Err(CKKSError::Internal(anyhow::anyhow!(
                        "PaCo module degree {} does not match plan degree {}",
                        module.n(),
                        plan.n()
                    )));
                }
                let mut owned = module.ckks_ciphertext_alloc_from_infos(ct);
                module.glwe_copy(&mut owned, ct, scratch);
                owned.set_meta_checked(ct.meta())?;
                Ok(::poulpy_ckks::encoding::paco_coeff_encodings_host::<$be, _, S>(
                    module, &owned, plan, base2k,
                )?)
            }
        }

        unsafe impl ::poulpy_ckks::oep::CKKSShipCoeffEncodingImpl for $be {
            fn ckks_ship_coeff_encodings_tmp_bytes_impl<S>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _plan: &::poulpy_ckks::layouts::ShipPlan,
                _base2k: ::poulpy_core::layouts::Base2K,
                _complex: bool,
            ) -> CKKSResult<usize>
            where
                S: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<S>,
            {
                Ok(0)
            }

            fn ckks_ship_coeff_encodings_impl<S, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct: &Src,
                plan: &::poulpy_ckks::layouts::ShipPlan,
                base2k: ::poulpy_core::layouts::Base2K,
                complex: bool,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> CKKSResult<
                ::poulpy_ckks::layouts::ShipCoeffEncodings<
                    <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                    <$be as ::poulpy_hal::layouts::Backend>::ZnxWord,
                >,
            >
            where
                S: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<S>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                use ::poulpy_ckks::layouts::CKKSModuleAlloc;
                use ::poulpy_core::GLWECopy;
                if module.n() != plan.n() {
                    return Err(CKKSError::Internal(anyhow::anyhow!(
                        "SHIP module degree {} does not match plan degree {}",
                        module.n(),
                        plan.n()
                    )));
                }
                let mut owned = module.ckks_ciphertext_alloc_from_infos(ct);
                module.glwe_copy(&mut owned, ct, scratch);
                owned.set_meta_checked(ct.meta())?;
                Ok(::poulpy_ckks::encoding::ship_coeff_encodings_host::<$be, _, S>(
                    module, &owned, plan, base2k, complex,
                )?)
            }
        }
    };
}

/// The CKKS compositions shared by both rings.
macro_rules! impl_oracle_ckks {
    ($be:ty) => {
        impl_oracle_ckks_encoding!($be);
        ::poulpy_ckks::impl_ckks_copy_reference!($be);
        ::poulpy_ckks::impl_ckks_encryption_reference!($be);
        ::poulpy_ckks::impl_ckks_mul_reference!($be);
        ::poulpy_ckks::impl_ckks_neg_reference!($be);
        ::poulpy_ckks::impl_ckks_pow2_reference!($be);
        ::poulpy_ckks::impl_ckks_rotate_reference!($be);
        ::poulpy_ckks::impl_ckks_add_reference!($be);
        ::poulpy_ckks::impl_ckks_sub_reference!($be);
        ::poulpy_ckks::impl_ckks_plaintext_reference!($be);
        ::poulpy_ckks::impl_ckks_polynomial_evaluation_reference!($be);
    };
}

/// The CKKS compositions of the standard ring, with bootstrapping.
macro_rules! impl_oracle_ckks_standard {
    ($be:ty) => {
        impl_oracle_ckks!($be);
        impl_oracle_ckks_coeff_encodings!($be);
        ::poulpy_ckks::impl_ckks_encapsulated_mod_up_reference!($be);
        ::poulpy_ckks::impl_ckks_conjugate_reference!($be);
        ::poulpy_ckks::impl_ckks_imag_reference!($be);
        ::poulpy_ckks::impl_ckks_bootstrapping_reference!($be);
        ::poulpy_ckks::impl_ckks_fold_reference!($be);
        ::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!($be);
        ::poulpy_ckks::impl_ckks_dft_reference!($be);
        ::poulpy_ckks::impl_ckks_eval_mod_reference!($be);
    };
}

impl_oracle_ckks_standard!(FFT64Oracle);
impl_oracle_ckks_standard!(NTT4x30Oracle);
impl_oracle_ckks!(FFT64CIOracle);
impl_oracle_ckks!(NTT4x30CIOracle);
