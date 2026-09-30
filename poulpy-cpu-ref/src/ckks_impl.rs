use super::{FFT64Ref, NTT4x30Ref};

use crate::ckks_encoding::CKKSEncodingTransform;
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;
use poulpy_hal::layouts::Ring;

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
// The reference backends have no accelerated transform, so they select the
// generic scalar table for every precision at once.
impl<R: Ring, F> CKKSEncodingTransform<F> for FFT64Ref<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

impl<R: Ring, F> CKKSEncodingTransform<F> for NTT4x30Ref<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

crate::impl_cpu_ckks_defaults!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_imag_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_fold_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_dft_reference!(super::FFT64Ref);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::FFT64Ref);
crate::impl_ckks_paco_coeff_encoding!(super::FFT64Ref);
crate::impl_ckks_ship_coeff_encoding!(super::FFT64Ref);
crate::impl_cpu_ckks_defaults!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_imag_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_fold_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_dft_reference!(super::NTT4x30Ref);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::NTT4x30Ref);
crate::impl_ckks_paco_coeff_encoding!(super::NTT4x30Ref);
crate::impl_ckks_ship_coeff_encoding!(super::NTT4x30Ref);
crate::impl_cpu_ckks_defaults!(super::FFT64CIRef);
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::FFT64CIRef);
crate::impl_cpu_ckks_defaults!(super::NTT4x30CIRef);
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::NTT4x30CIRef);
