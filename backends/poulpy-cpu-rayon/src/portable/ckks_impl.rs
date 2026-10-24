//! `poulpy-ckks` extension points of the portable Rayon backends.

use poulpy_ckks::{impl_ckks_encapsulated_mod_up_reference_full, impl_ckks_mod_up_first_factor_reference_full};
use poulpy_cpu_portable::ckks_encoding::{CKKSEncodingTransform, EncodingFFTTable};
use poulpy_hal::layouts::Ring;

use super::{FFT64PortableRayon, NTT4x30PortableRayon};

impl_ckks_encapsulated_mod_up_reference_full!(FFT64PortableRayon);
impl_ckks_encapsulated_mod_up_reference_full!(NTT4x30PortableRayon);
impl_ckks_mod_up_first_factor_reference_full!(FFT64PortableRayon);
impl_ckks_mod_up_first_factor_reference_full!(NTT4x30PortableRayon);

// The portable backends encode every precision with the canonical table.
impl<R: Ring, F> CKKSEncodingTransform<F> for FFT64PortableRayon<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = EncodingFFTTable<F>;
}

impl<R: Ring, F> CKKSEncodingTransform<F> for NTT4x30PortableRayon<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = EncodingFFTTable<F>;
}

poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_conjugate_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_imag_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_ci_reference_full!(super::FFT64PortableRayon, super::FFT64CIPortableRayon);
::poulpy_ckks::impl_ckks_fold_layout_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_fold_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_dft_reference_full!(super::FFT64PortableRayon);
::poulpy_ckks::impl_dft_matrix_reference_full!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64PortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_conjugate_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_imag_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_bootstrapping_ci_reference_full!(super::NTT4x30PortableRayon, super::NTT4x30CIPortableRayon);
::poulpy_ckks::impl_ckks_fold_layout_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_fold_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_dft_reference_full!(super::NTT4x30PortableRayon);
::poulpy_ckks::impl_dft_matrix_reference_full!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30PortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CIPortableRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CIPortableRayon);
