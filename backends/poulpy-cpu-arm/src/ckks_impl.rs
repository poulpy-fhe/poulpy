use super::{FFT64Neon, NTT4x30Neon};
use poulpy_ckks::{
    impl_ckks_bootstrapping_ci_reference_full, impl_ckks_bootstrapping_reference_full,
    impl_ckks_complex_polynomial_evaluation_reference_full, impl_ckks_conjugate_reference_full,
    impl_ckks_encapsulated_mod_up_reference_full, impl_ckks_fold_layout_reference_full, impl_ckks_fold_reference_full,
    impl_ckks_imag_reference_full, impl_ckks_mod_up_first_factor_reference_full, impl_ckks_paco_derived_full,
    impl_ckks_ring_map_derived_full, impl_ckks_ship_reference_full, impl_dft_matrix_reference_full, impl_dft_reference_full,
};

impl_ckks_encapsulated_mod_up_reference_full!(FFT64Neon);
impl_ckks_mod_up_first_factor_reference_full!(FFT64Neon);
impl_ckks_encapsulated_mod_up_reference_full!(NTT4x30Neon);
impl_ckks_mod_up_first_factor_reference_full!(NTT4x30Neon);
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
impl_ckks_conjugate_reference_full!(super::FFT64Neon);
impl_ckks_imag_reference_full!(super::FFT64Neon);
impl_ckks_bootstrapping_reference_full!(super::FFT64Neon);
impl_ckks_bootstrapping_ci_reference_full!(super::FFT64Neon, super::FFT64CINeon);
impl_ckks_fold_layout_reference_full!(super::FFT64Neon);
impl_ckks_fold_reference_full!(super::FFT64Neon);
impl_ckks_complex_polynomial_evaluation_reference_full!(super::FFT64Neon);
impl_dft_reference_full!(super::FFT64Neon);
impl_dft_matrix_reference_full!(super::FFT64Neon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64Neon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64Neon);
impl_ckks_ring_map_derived_full!(super::FFT64Neon);
impl_ckks_paco_derived_full!(super::FFT64Neon);
impl_ckks_ship_reference_full!(super::FFT64Neon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30Neon);
impl_ckks_conjugate_reference_full!(super::NTT4x30Neon);
impl_ckks_imag_reference_full!(super::NTT4x30Neon);
impl_ckks_bootstrapping_reference_full!(super::NTT4x30Neon);
impl_ckks_bootstrapping_ci_reference_full!(super::NTT4x30Neon, super::NTT4x30CINeon);
impl_ckks_fold_layout_reference_full!(super::NTT4x30Neon);
impl_ckks_fold_reference_full!(super::NTT4x30Neon);
impl_ckks_complex_polynomial_evaluation_reference_full!(super::NTT4x30Neon);
impl_dft_reference_full!(super::NTT4x30Neon);
impl_dft_matrix_reference_full!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30Neon);
impl_ckks_ring_map_derived_full!(super::NTT4x30Neon);
impl_ckks_paco_derived_full!(super::NTT4x30Neon);
impl_ckks_ship_reference_full!(super::NTT4x30Neon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_ci_reference_full!(super::FFT64NeonRayon, super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_layout_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_dft_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_dft_matrix_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ring_map_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_paco_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ship_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_ci_reference_full!(super::NTT4x30NeonRayon, super::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_layout_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_dft_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_dft_matrix_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_paco_coeff_encoding!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_ckks_ship_coeff_encoding!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ring_map_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_paco_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_ship_reference_full!(super::NTT4x30NeonRayon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CINeon);
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CINeon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_ckks_defaults!(super::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_mod_up_first_factor_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_mod_up_first_factor_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(super::NTT4x30CINeonRayon);
