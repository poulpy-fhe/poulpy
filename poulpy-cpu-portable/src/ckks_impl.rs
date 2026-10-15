use super::{FFT64Portable, NTT4x30Portable};

use crate::ckks_encoding::CKKSEncodingTransform;
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;
use poulpy_hal::layouts::Ring;

impl_ckks_encapsulated_mod_up_reference!(FFT64Portable);
// The portable backends encode every precision with the canonical table.
impl<R: Ring, F> CKKSEncodingTransform<F> for FFT64Portable<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::ckks_encoding::EncodingFFTTable<F>;
}

impl<R: Ring, F> CKKSEncodingTransform<F> for NTT4x30Portable<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::ckks_encoding::EncodingFFTTable<F>;
}

crate::impl_cpu_ckks_defaults!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_imag_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_fold_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_dft_reference!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::FFT64Portable);
crate::impl_ckks_paco_coeff_encoding!(super::FFT64Portable);
crate::impl_ckks_ship_coeff_encoding!(super::FFT64Portable);
crate::impl_cpu_ckks_defaults!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_imag_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_fold_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_dft_reference!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::NTT4x30Portable);
crate::impl_ckks_paco_coeff_encoding!(super::NTT4x30Portable);
crate::impl_ckks_ship_coeff_encoding!(super::NTT4x30Portable);
crate::impl_cpu_ckks_defaults!(super::FFT64CIPortable);
crate::impl_cpu_ckks_defaults!(super::NTT4x30CIPortable);
