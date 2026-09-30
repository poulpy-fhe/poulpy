use super::{FFT64Neon, NTT4x30Neon};
use poulpy_ckks::{
    impl_ckks_bootstrapping_reference, impl_ckks_ci_ring_map_reference, impl_ckks_complex_polynomial_evaluation_reference,
    impl_ckks_conjugate_reference, impl_ckks_dft_reference, impl_ckks_encapsulated_mod_up_reference,
    impl_ckks_eval_mod_reference, impl_ckks_fold_reference, impl_ckks_imag_reference, oep::CIBridge,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Neon);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Neon);
// `f64` encodes through the NEON kernels and `Quad` through the portable
// kernels, both on the canonical twiddles. Rust has no specialization, so
// accelerated backends list their precisions explicitly.
macro_rules! select_neon_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_portable::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = super::FFT64NeonEncodingTable;
        }

        impl ::poulpy_cpu_portable::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<poulpy_ckks::Quad>;
        }
    };
}

select_neon_encoding_transform!(FFT64Neon);
select_neon_encoding_transform!(NTT4x30Neon);
select_neon_encoding_transform!(super::FFT64CINeon);
select_neon_encoding_transform!(super::NTT4x30CINeon);

poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64Neon);
impl_ckks_conjugate_reference!(super::FFT64Neon);
impl_ckks_imag_reference!(super::FFT64Neon);
impl_ckks_bootstrapping_reference!(super::FFT64Neon);
impl_ckks_fold_reference!(super::FFT64Neon);
impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Neon);
impl_ckks_dft_reference!(super::FFT64Neon);
impl_ckks_eval_mod_reference!(super::FFT64Neon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64Neon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64Neon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30Neon);
impl_ckks_conjugate_reference!(super::NTT4x30Neon);
impl_ckks_imag_reference!(super::NTT4x30Neon);
impl_ckks_bootstrapping_reference!(super::NTT4x30Neon);
impl_ckks_fold_reference!(super::NTT4x30Neon);
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Neon);
impl_ckks_dft_reference!(super::NTT4x30Neon);
impl_ckks_eval_mod_reference!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30Neon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_eval_mod_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_eval_mod_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30NeonRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CINeon);
impl_ckks_ci_ring_map_reference!(super::FFT64CINeon);
impl CIBridge for super::FFT64Neon {
    type CI = super::FFT64CINeon;
}
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CINeon);
impl_ckks_ci_ring_map_reference!(super::NTT4x30CINeon);
impl CIBridge for super::NTT4x30Neon {
    type CI = super::NTT4x30CINeon;
}
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ci_ring_map_reference!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl CIBridge for super::FFT64NeonRayon {
    type CI = super::FFT64CINeonRayon;
}
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ci_ring_map_reference!(super::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl CIBridge for super::NTT4x30NeonRayon {
    type CI = super::NTT4x30CINeonRayon;
}
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::NTT4x30CINeonRayon);
