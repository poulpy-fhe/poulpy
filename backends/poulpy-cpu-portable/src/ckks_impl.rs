use super::{FFT64Portable, NTT4x30Portable};

use crate::ckks_encoding::CKKSEncodingTransform;
use poulpy_ckks::{impl_ckks_encapsulated_mod_up_reference_full, impl_ckks_mod_up_first_factor_reference_full};
use poulpy_hal::layouts::Ring;

impl_ckks_encapsulated_mod_up_reference_full!(FFT64Portable);
impl_ckks_encapsulated_mod_up_reference_full!(NTT4x30Portable);
impl_ckks_mod_up_first_factor_reference_full!(FFT64Portable);
impl_ckks_mod_up_first_factor_reference_full!(NTT4x30Portable);
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
::poulpy_ckks::impl_ckks_conjugate_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_imag_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_bootstrapping_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_bootstrapping_ci_reference_full!(super::FFT64Portable, super::FFT64CIPortable);
::poulpy_ckks::impl_ckks_fold_layout_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_fold_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_dft_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_dft_matrix_reference_full!(super::FFT64Portable);
crate::impl_ckks_paco_coeff_encoding!(super::FFT64Portable);
crate::impl_ckks_ship_coeff_encoding!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_ci_ring_map_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_paco_reference_full!(super::FFT64Portable);
::poulpy_ckks::impl_ckks_ship_reference_full!(super::FFT64Portable);
crate::impl_cpu_ckks_defaults!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_conjugate_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_imag_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_bootstrapping_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_bootstrapping_ci_reference_full!(super::NTT4x30Portable, super::NTT4x30CIPortable);
::poulpy_ckks::impl_ckks_fold_layout_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_fold_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_dft_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_dft_matrix_reference_full!(super::NTT4x30Portable);
crate::impl_ckks_paco_coeff_encoding!(super::NTT4x30Portable);
crate::impl_ckks_ship_coeff_encoding!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_ci_ring_map_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_paco_reference_full!(super::NTT4x30Portable);
::poulpy_ckks::impl_ckks_ship_reference_full!(super::NTT4x30Portable);
crate::impl_cpu_ckks_defaults!(super::FFT64CIPortable);
crate::impl_cpu_ckks_defaults!(super::NTT4x30CIPortable);
