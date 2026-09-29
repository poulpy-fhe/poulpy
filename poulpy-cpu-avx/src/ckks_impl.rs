use super::{FFT64Avx, FFT64CIAvx, NTT4x30Avx, NTT4x30CIAvx};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx);
// `f64` encodes through the AVX2/FMA kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = super::FFT64AvxReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx_encoding_transform!(FFT64Avx);
select_avx_encoding_transform!(NTT4x30Avx);
select_avx_encoding_transform!(FFT64CIAvx);
select_avx_encoding_transform!(NTT4x30CIAvx);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_imag_reference!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_dft_reference!(super::FFT64Avx);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::FFT64Avx);
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::FFT64Avx);
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::FFT64Avx);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_conjugate_reference!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_imag_reference!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_dft_reference!(super::NTT4x30Avx);
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::NTT4x30Avx);
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT4x30Avx);
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT4x30Avx);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64CIAvx);
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::FFT64CIAvx);
impl ::poulpy_ckks::oep::CIBridge for super::FFT64Avx {
    type CI = super::FFT64CIAvx;
}
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30CIAvx);
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::NTT4x30CIAvx);
impl ::poulpy_ckks::oep::CIBridge for super::NTT4x30Avx {
    type CI = super::NTT4x30CIAvx;
}
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_conjugate_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_imag_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_dft_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_conjugate_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_imag_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_bootstrapping_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_dft_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_eval_mod_reference!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
impl ::poulpy_ckks::oep::CIBridge for super::FFT64AvxRayon {
    type CI = super::FFT64CIAvxRayon;
}
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30CIAvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_ckks::impl_ckks_ci_ring_map_reference!(super::NTT4x30CIAvxRayon);
#[cfg(feature = "enable-rayon")]
impl ::poulpy_ckks::oep::CIBridge for super::NTT4x30AvxRayon {
    type CI = super::NTT4x30CIAvxRayon;
}
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(super::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(super::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(super::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(super::NTT4x30CIAvxRayon);
