//! CPU implementation of the CKKS slot-encoding extension point.
//!
//! The encoder is written once and is generic over the negacyclic transform:
//! slot permutation and quantization come from `poulpy-ckks` reference routines.
//! This module owns the transform families and their cache bindings.
//! A backend selects its transform by implementing [`CKKSEncodingTransform`]
//! for each scalar precision it supports, and instantiates the encoder with
//! [`impl_ckks_encoding!`](crate::impl_ckks_encoding).
//!
//! A plan-cache entry holds the complete geometric family for one scalar
//! precision. The encoder owns its twiddle tables rather than borrowing the
//! backend's ring-FFT tables: the duplication is a geometric series bounded by
//! `2·max_n` scalars (~100 KiB of `f64` at `n = 65536`), and in exchange every
//! backend follows one code path and can use its own accelerated kernels.

use std::marker::PhantomData;

use anyhow::{Result, ensure};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use poulpy_ckks::reference::encoding::EncodingPermutation;

/// Precision-independent slot maps for all powers of two up to `max_slots`.
pub struct EncodingPlanSet {
    plans: Vec<EncodingPermutation>,
    max_slots: usize,
}

impl EncodingPlanSet {
    pub fn new(max_slots: usize) -> Result<Self> {
        ensure!(
            max_slots > 0 && max_slots.is_power_of_two(),
            "maximum slot count must be a non-zero power of two"
        );
        let plans = (0..=max_slots.ilog2() as usize)
            .map(|log_slots| EncodingPermutation::new(1usize << log_slots))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { plans, max_slots })
    }

    pub fn for_slots(&self, slots: usize) -> Result<&EncodingPermutation> {
        ensure!(
            slots > 0 && slots.is_power_of_two(),
            "slot count must be a non-zero power of two, got {slots}"
        );
        ensure!(
            slots <= self.max_slots,
            "slot count {slots} exceeds module capacity {}",
            self.max_slots
        );
        Ok(&self.plans[slots.ilog2() as usize])
    }
}

/// Complete geometric family of negacyclic transforms, one per power-of-two
/// slot count up to `max_slots`.
///
/// The family is owned by the encoder rather than borrowed from the ring
/// backend, so a backend may pick a transform whose kernels differ from the
/// ones its ring operations use.
pub struct NegacyclicFFTSet<F, T> {
    ffts: Vec<T>,
    max_slots: usize,
    scalar: PhantomData<F>,
}

impl<F, T> NegacyclicFFTSet<F, T>
where
    T: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    pub fn new(max_slots: usize) -> Result<Self> {
        ensure!(
            max_slots > 0 && max_slots.is_power_of_two(),
            "maximum slot count must be a non-zero power of two"
        );
        // `NegacyclicFFTNew::new` takes the transform half-length, which for a
        // `slots`-slot encoding is `slots` itself.
        let ffts = (0..=max_slots.ilog2() as usize)
            .map(|log_slots| T::new(1usize << log_slots))
            .collect();
        Ok(Self {
            ffts,
            max_slots,
            scalar: PhantomData,
        })
    }

    pub fn for_slots(&self, slots: usize) -> Result<&T> {
        ensure!(
            slots > 0 && slots.is_power_of_two(),
            "slot count must be a non-zero power of two, got {slots}"
        );
        ensure!(
            slots <= self.max_slots,
            "slot count {slots} exceeds module capacity {}",
            self.max_slots
        );
        Ok(&self.ffts[slots.ilog2() as usize])
    }
}

/// CKKS encoding plans for precision `F` over the transform `T`.
///
/// Pairs the precision-independent slot maps with the owned transform family.
pub struct OwnedEncodingPlanSet<F, T> {
    maps: EncodingPlanSet,
    ffts: NegacyclicFFTSet<F, T>,
}

impl<F, T> OwnedEncodingPlanSet<F, T>
where
    T: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    /// Builds the family for every power-of-two slot count up to `max_n / 2`.
    pub fn new(max_n: usize) -> Result<Self> {
        let max_slots = max_n / 2;
        Ok(Self {
            maps: EncodingPlanSet::new(max_slots)?,
            ffts: NegacyclicFFTSet::new(max_slots)?,
        })
    }

    #[doc(hidden)]
    pub fn for_slots(&self, slots: usize) -> Result<(&EncodingPermutation, &T)> {
        Ok((self.maps.for_slots(slots)?, self.ffts.for_slots(slots)?))
    }
}

/// Selects the negacyclic transform a backend uses to encode at precision `F`.
///
/// This is the single point a backend crate customizes: the encoder itself is
/// written once, generically, in this module. A backend with an accelerated
/// transform implements this once per precision it accelerates; a backend
/// without one may implement it blanketly over every `F`, which is what makes
/// higher precisions such as `Quad` work with no extra code.
///
/// Rust has no specialization, so a blanket implementation and a
/// precision-specific one cannot coexist for the same backend: accelerated
/// backends enumerate their precisions explicitly (one line each).
pub trait CKKSEncodingTransform<F> {
    /// The transform family used for precision `F`.
    type Fft: NegacyclicFFT<F> + NegacyclicFFTNew<F> + Send + Sync + 'static;
}

/// Instantiates the generic CKKS encoder for a backend.
///
/// The scheme math comes from the CKKS reference helpers; the backend supplies
/// its transform through [`CKKSEncodingTransform`](crate::ckks_encoding::CKKSEncodingTransform),
/// so this expands to a single implementation covering every precision that
/// backend selects a transform for.
#[macro_export]
macro_rules! impl_ckks_encoding {
    ($be:ty) => {
        unsafe impl<F> ::poulpy_ckks::oep::CKKSEncodingImpl<F> for $be
        where
            F: ::poulpy_ckks::api::CKKSEncodingScalar,
            $be: $crate::ckks_encoding::CKKSEncodingTransform<F>,
        {
            type Plans =
                $crate::ckks_encoding::OwnedEncodingPlanSet<F, <$be as $crate::ckks_encoding::CKKSEncodingTransform<F>>::Fft>;

            fn ckks_encoding_plan_cache_impl(
                module: &::poulpy_hal::layouts::Module<$be>,
            ) -> &::poulpy_hal::layouts::ModulePlanCache {
                $crate::table_cache::ModuleTableCacheAccess::module_table_cache(module)
            }

            fn ckks_encoding_plans_create_impl(
                module: &::poulpy_hal::layouts::Module<$be>,
            ) -> ::poulpy_ckks::CKKSResult<Self::Plans> {
                $crate::ckks_encoding::OwnedEncodingPlanSet::new(module.max_n()).map_err(::poulpy_ckks::CKKSError::from)
            }

            fn ckks_encode_coeffs_into_impl<P>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                pt: &mut P,
                coeffs: &::poulpy_ckks::layouts::CKKSEncodingBufferBackendRef<'_, $be, F>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                P: ::poulpy_ckks::CKKSPlaintextToBackendMut<$be> + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::encoding::encode_coeffs_into_host::<$be, F, P>(pt, coeffs)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }

            fn ckks_decode_coeffs_into_impl<P>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                pt: &P,
                coeffs: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, F>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                P: ::poulpy_ckks::CKKSPlaintextToBackendRef<$be> + ::poulpy_core::layouts::IntPolyInfos,
            {
                ::poulpy_ckks::reference::encoding::decode_coeffs_into_host::<$be, F, P>(pt, coeffs)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }

            fn ckks_slots_to_coeffs_assign_impl(
                _module: &::poulpy_hal::layouts::Module<$be>,
                plans: &Self::Plans,
                values: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, F>,
            ) -> ::poulpy_ckks::CKKSResult<()> {
                let slots = ::poulpy_ckks::layouts::CKKSEncodingBufferInfos::len(values) / 2;
                let (map, fft) = plans.for_slots(slots)?;
                map.slots_to_coeffs_assign(fft, values.as_mut_slice())
                    .map_err(::poulpy_ckks::CKKSError::from)
            }

            fn ckks_coeffs_to_slots_assign_impl(
                _module: &::poulpy_hal::layouts::Module<$be>,
                plans: &Self::Plans,
                values: &mut ::poulpy_ckks::layouts::CKKSEncodingBufferBackendMut<'_, $be, F>,
            ) -> ::poulpy_ckks::CKKSResult<()> {
                let slots = ::poulpy_ckks::layouts::CKKSEncodingBufferInfos::len(values) / 2;
                let (map, fft) = plans.for_slots(slots)?;
                map.coeffs_to_slots_assign(fft, values.as_mut_slice())
                    .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{FFT64Ref, NTT4x30Ref};
    use poulpy_ckks::{
        CKKSMeta, SetCKKSInfos, SlotsKind,
        api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
        layouts::{CKKSEncodingBuffer, CKKSModuleAlloc, ScratchArenaTakeCKKS},
    };
    use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, TorusPrecision};
    use poulpy_hal::{
        AlignedBuf,
        api::ScratchOwnedAlloc,
        layouts::{Backend, Module, ScratchOwned},
    };

    fn assert_close<F: CKKSEncodingScalar>(got_re: &[F], got_im: &[F], re: &[F], im: &[F]) {
        for (got, want) in got_re.iter().chain(got_im).zip(re.iter().chain(im)) {
            assert!((got.to_f64().unwrap() - want.to_f64().unwrap()).abs() < 1e-8);
        }
    }

    fn roundtrip_all_dimensions<BE, F>(module: &Module<BE>)
    where
        BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
        F: CKKSEncodingScalar,
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    {
        for slots in [1usize, 2, 4, 8, 16] {
            let re: Vec<F> = (0..slots).map(|i| F::from_f64((i as f64 + 1.0) / 17.0).unwrap()).collect();
            let im: Vec<F> = (0..slots).map(|i| F::from_f64(-(i as f64 + 1.0) / 29.0).unwrap()).collect();
            let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
            let mut scratch = scratch.arena();

            for (degree, log_sparsity) in [
                (2 * slots, 0),
                (module.max_n(), (module.max_n() / (2 * slots)).ilog2() as usize),
            ] {
                let mut pt = module.ckks_plaintext_alloc(Degree(degree as u32), Base2K(20), TorusPrecision(80));
                pt.set_meta(CKKSMeta {
                    log_sparsity,
                    log_delta: 40,
                    slots: SlotsKind::Complex,
                });
                module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch).unwrap();
                let mut got_re = vec![F::zero(); slots];
                let mut got_im = vec![F::zero(); slots];
                module
                    .ckks_decode_reim_into(&pt, &mut got_re, &mut got_im, &mut scratch)
                    .unwrap();
                assert_close(&got_re, &got_im, &re, &im);
            }
        }
    }

    #[test]
    fn fft64_f64_reuses_ring_plan_family_at_every_dimension() {
        let module = Module::<FFT64Ref>::new(32);
        roundtrip_all_dimensions::<FFT64Ref, f64>(&module);
    }

    #[test]
    fn fft64_precision_families_coexist_in_one_module() {
        let module = Module::<FFT64Ref>::new(32);
        roundtrip_all_dimensions::<FFT64Ref, f64>(&module);
        roundtrip_all_dimensions::<FFT64Ref, poulpy_ckks::Quad>(&module);
    }

    #[test]
    fn ntt_encoding_precisions_coexist_independently_of_ring_transform() {
        let module = Module::<NTT4x30Ref>::new(32);
        roundtrip_all_dimensions::<NTT4x30Ref, f64>(&module);
        roundtrip_all_dimensions::<NTT4x30Ref, poulpy_ckks::Quad>(&module);
    }

    #[test]
    fn coefficient_only_encoding_accepts_scratch_plaintext_views() {
        let module = Module::<FFT64Ref>::new(32);
        let mut layout = module.ckks_plaintext_alloc(Degree(32), Base2K(20), TorusPrecision(80));
        layout.set_meta(CKKSMeta {
            log_sparsity: 1,
            log_delta: 40,
            slots: SlotsKind::Complex,
        });
        let bytes = GLWEPlaintext::<AlignedBuf, i64>::bytes_of_from_infos(&layout);
        let mut pt_scratch = ScratchOwned::<FFT64Ref>::alloc(bytes);
        let (mut pt, _) = pt_scratch.arena().take_ckks_plaintext_like_scratch(&layout);

        let slots = 8;
        let re: Vec<f64> = (0..slots).map(|i| (i as f64 + 1.0) / 17.0).collect();
        let im: Vec<f64> = (0..slots).map(|i| -(i as f64 + 1.0) / 29.0).collect();
        let values = re.iter().chain(&im).copied().collect::<Vec<_>>();
        let mut values = CKKSEncodingBuffer::<AlignedBuf, f64>::from_host::<FFT64Ref>(&values);
        module.ckks_encode_slots_assign_into(&mut pt, &mut values).unwrap();
        let coeffs = values.to_host::<FFT64Ref>();

        let mut decoded = CKKSEncodingBuffer::<AlignedBuf, f64>::from_host::<FFT64Ref>(&vec![0.0; 2 * slots]);
        module.ckks_decode_slots_into(&pt, &mut decoded).unwrap();
        let values = decoded.to_host::<FFT64Ref>();
        let (got_re, got_im) = values.split_at(slots);
        assert_close(got_re, got_im, &re, &im);

        let mut coeff_scratch =
            ScratchOwned::<FFT64Ref>::alloc(CKKSEncodingHostOps::<FFT64Ref, f64>::ckks_reim_tmp_bytes(&module, slots));
        let mut coeff_scratch = coeff_scratch.arena();
        module
            .ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut coeff_scratch)
            .unwrap();
        let mut got_coeffs = vec![0.0; coeffs.len()];
        module
            .ckks_decode_coeffs_host_into(&pt, &mut got_coeffs, &mut coeff_scratch)
            .unwrap();
        for (got, want) in got_coeffs.iter().zip(&coeffs) {
            assert!((got - want).abs() < 1e-8);
        }

        let mut reim_scratch =
            ScratchOwned::<FFT64Ref>::alloc(CKKSEncodingHostOps::<FFT64Ref, f64>::ckks_reim_tmp_bytes(&module, slots));
        let mut reim_scratch = reim_scratch.arena();
        module.ckks_encode_reim_into(&mut pt, &re, &im, &mut reim_scratch).unwrap();
        let mut got_re = vec![0.0; slots];
        let mut got_im = vec![0.0; slots];
        module
            .ckks_decode_reim_into(&pt, &mut got_re, &mut got_im, &mut reim_scratch)
            .unwrap();
        assert_close(&got_re, &got_im, &re, &im);
    }

    #[test]
    fn encoding_rejects_invalid_slot_shapes() {
        let module = Module::<FFT64Ref>::new(32);
        let mut pt = module.ckks_plaintext_alloc(Degree(32), Base2K(20), TorusPrecision(80));
        pt.set_meta(CKKSMeta {
            log_sparsity: 0,
            log_delta: 40,
            slots: SlotsKind::Complex,
        });

        let three = [0.0; 3];
        let mut scratch = ScratchOwned::<FFT64Ref>::alloc(CKKSEncodingHostOps::<FFT64Ref, f64>::ckks_reim_tmp_bytes(&module, 8));
        let mut scratch = scratch.arena();
        assert!(module.ckks_encode_reim_into(&mut pt, &three, &three, &mut scratch).is_err());

        let re = [0.0; 8];
        let im = [0.0; 7];
        assert!(module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch).is_err());

        let im = [0.0; 8];
        assert!(module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch).is_ok());

        let required = CKKSEncodingHostOps::<FFT64Ref, f64>::ckks_reim_tmp_bytes(&module, 8);
        let mut undersized = ScratchOwned::<FFT64Ref>::alloc(required - FFT64Ref::SCRATCH_ALIGN);
        assert!(
            module
                .ckks_encode_reim_into(&mut pt, &re, &im, &mut undersized.arena())
                .is_err()
        );
    }
}
