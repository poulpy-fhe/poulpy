#[cfg(feature = "enable-rayon")]
use super::FFT64Avx512Rayon;
#[cfg(feature = "enable-ifma")]
use super::NTT3x42Ifma;
use super::{FFT64Avx512, NTT4x30Avx512};
use poulpy_ckks::{
    impl_ckks_bootstrapping_reference, impl_ckks_complex_polynomial_evaluation_reference, impl_ckks_conjugate_reference,
    impl_ckks_dft_reference, impl_ckks_encapsulated_mod_up_reference, impl_ckks_eval_mod_reference, impl_ckks_fold_reference,
    impl_ckks_imag_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512Rayon);

// `f64` encodes through the AVX-512 kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx512_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = super::FFT64Avx512ReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx512_encoding_transform!(FFT64Avx512);
select_avx512_encoding_transform!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(NTT3x42Ifma);

#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(FFT64Avx512Rayon);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64Avx512);
impl_ckks_conjugate_reference!(super::FFT64Avx512);
impl_ckks_imag_reference!(super::FFT64Avx512);
impl_ckks_bootstrapping_reference!(super::FFT64Avx512);
impl_ckks_fold_reference!(super::FFT64Avx512);
impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Avx512);
impl_ckks_dft_reference!(super::FFT64Avx512);
impl_ckks_eval_mod_reference!(super::FFT64Avx512);
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::FFT64Avx512);
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::FFT64Avx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30Avx512);
impl_ckks_conjugate_reference!(super::NTT4x30Avx512);
impl_ckks_imag_reference!(super::NTT4x30Avx512);
impl_ckks_bootstrapping_reference!(super::NTT4x30Avx512);
impl_ckks_fold_reference!(super::NTT4x30Avx512);
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Avx512);
impl_ckks_dft_reference!(super::NTT4x30Avx512);
impl_ckks_eval_mod_reference!(super::NTT4x30Avx512);
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT4x30Avx512);
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_conjugate_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_imag_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_bootstrapping_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_fold_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
#[cfg(feature = "enable-ifma")]
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_dft_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_eval_mod_reference!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT3x42Ifma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_eval_mod_reference!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_bootstrapping_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_fold_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
#[cfg(feature = "enable-rayon")]
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_eval_mod_reference!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(
    super::NTT3x42IfmaRayon,
    prepared_tensor = crate::core_impl::ifma_prepared_tensor
);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_conjugate_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_imag_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_bootstrapping_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_fold_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_complex_polynomial_evaluation_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_dft_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
impl_ckks_eval_mod_reference!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(super::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(super::NTT3x42IfmaRayon);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64CIAvx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT3x42CIIfma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30CIAvx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT3x42CIIfmaRayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
select_avx512_encoding_transform!(super::NTT3x42IfmaRayon);
select_avx512_encoding_transform!(super::FFT64CIAvx512);
select_avx512_encoding_transform!(super::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(super::NTT3x42CIIfma);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(super::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(super::NTT4x30CIAvx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
select_avx512_encoding_transform!(super::NTT3x42CIIfmaRayon);
