#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
use crate::NTT3x42IfmaRayon;
use crate::{FFT64Avx512, NTT4x30Avx512};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64Avx512Rayon, NTT4x30Avx512Rayon};
use poulpy_ckks::{
    impl_ckks_add_reference, impl_ckks_conjugate_reference, impl_ckks_copy_reference, impl_ckks_dft_reference,
    impl_ckks_encapsulated_mod_up_reference, impl_ckks_encryption_reference, impl_ckks_imag_reference, impl_ckks_mul_reference,
    impl_ckks_neg_reference, impl_ckks_plaintext_reference, impl_ckks_pow2_reference, impl_ckks_rotate_reference,
    impl_ckks_sub_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512);
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvx512);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvx512Rayon);

impl_ckks_conjugate_reference!(FFT64Avx512);
impl_ckks_conjugate_reference!(crate::FFT64CIAvx512);
impl_ckks_conjugate_reference!(NTT4x30Avx512);
impl_ckks_conjugate_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_conjugate_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_conjugate_reference!(crate::NTT3x42CIIfma);

impl_ckks_copy_reference!(FFT64Avx512);
impl_ckks_copy_reference!(crate::FFT64CIAvx512);
impl_ckks_copy_reference!(NTT4x30Avx512);
impl_ckks_copy_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_copy_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_copy_reference!(crate::NTT3x42CIIfma);

impl_ckks_encryption_reference!(FFT64Avx512);
impl_ckks_encryption_reference!(crate::FFT64CIAvx512);
impl_ckks_encryption_reference!(NTT4x30Avx512);
impl_ckks_encryption_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_encryption_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_encryption_reference!(crate::NTT3x42CIIfma);

impl_ckks_imag_reference!(FFT64Avx512);
impl_ckks_imag_reference!(crate::FFT64CIAvx512);
impl_ckks_imag_reference!(NTT4x30Avx512);
impl_ckks_imag_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_imag_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_imag_reference!(crate::NTT3x42CIIfma);

impl_ckks_mul_reference!(FFT64Avx512);
impl_ckks_mul_reference!(crate::FFT64CIAvx512);
impl_ckks_mul_reference!(NTT4x30Avx512);
impl_ckks_mul_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_mul_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_mul_reference!(crate::NTT3x42CIIfma);

impl_ckks_neg_reference!(FFT64Avx512);
impl_ckks_neg_reference!(crate::FFT64CIAvx512);
impl_ckks_neg_reference!(NTT4x30Avx512);
impl_ckks_neg_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_neg_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_neg_reference!(crate::NTT3x42CIIfma);

impl_ckks_pow2_reference!(FFT64Avx512);
impl_ckks_pow2_reference!(crate::FFT64CIAvx512);
impl_ckks_pow2_reference!(NTT4x30Avx512);
impl_ckks_pow2_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_pow2_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_pow2_reference!(crate::NTT3x42CIIfma);

impl_ckks_rotate_reference!(FFT64Avx512);
impl_ckks_rotate_reference!(crate::FFT64CIAvx512);
impl_ckks_rotate_reference!(NTT4x30Avx512);
impl_ckks_rotate_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_rotate_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_rotate_reference!(crate::NTT3x42CIIfma);

// `f64` encodes through the AVX-512 kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx512_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = crate::FFT64Avx512ReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx512_encoding_transform!(FFT64Avx512);
select_avx512_encoding_transform!(crate::FFT64CIAvx512);
select_avx512_encoding_transform!(NTT4x30Avx512);
select_avx512_encoding_transform!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(crate::NTT3x42CIIfma);

::poulpy_cpu_ref::impl_ckks_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_encoding!(crate::FFT64CIAvx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::FFT64CIAvx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::FFT64CIAvx512);
::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30Avx512);
::poulpy_cpu_ref::impl_ckks_encoding!(crate::NTT4x30CIAvx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30Avx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::NTT4x30CIAvx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30Avx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_encoding!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_encoding!(crate::NTT3x42CIIfma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::NTT3x42CIIfma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::NTT3x42CIIfma);

impl_ckks_add_reference!(FFT64Avx512);
impl_ckks_add_reference!(crate::FFT64CIAvx512);
impl_ckks_add_reference!(NTT4x30Avx512);
impl_ckks_add_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_add_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_add_reference!(crate::NTT3x42CIIfma);

impl_ckks_sub_reference!(FFT64Avx512);
impl_ckks_sub_reference!(crate::FFT64CIAvx512);
impl_ckks_sub_reference!(NTT4x30Avx512);
impl_ckks_sub_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_sub_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_sub_reference!(crate::NTT3x42CIIfma);

impl_ckks_plaintext_reference!(FFT64Avx512);
impl_ckks_plaintext_reference!(crate::FFT64CIAvx512);
impl_ckks_plaintext_reference!(NTT4x30Avx512);
impl_ckks_plaintext_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_plaintext_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_plaintext_reference!(crate::NTT3x42CIIfma);

impl_ckks_dft_reference!(FFT64Avx512);
impl_ckks_dft_reference!(crate::FFT64CIAvx512);
impl_ckks_dft_reference!(NTT4x30Avx512);
impl_ckks_dft_reference!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_dft_reference!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
impl_ckks_dft_reference!(crate::NTT3x42CIIfma);

#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_conjugate_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_copy_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_copy_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encryption_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encryption_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_imag_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_mul_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_mul_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_neg_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_neg_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_pow2_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_pow2_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_rotate_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_rotate_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_encoding!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_encoding!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_add_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_add_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_sub_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_sub_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_plaintext_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_plaintext_reference!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_dft_reference!(crate::FFT64CIAvx512Rayon);

#[cfg(feature = "enable-rayon")]
mod ntt4x30_rayon_defaults {
    use super::*;

    impl_ckks_conjugate_reference!(NTT4x30Avx512Rayon);
    impl_ckks_conjugate_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_copy_reference!(NTT4x30Avx512Rayon);
    impl_ckks_copy_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_encryption_reference!(NTT4x30Avx512Rayon);
    impl_ckks_encryption_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_imag_reference!(NTT4x30Avx512Rayon);
    impl_ckks_imag_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_mul_reference!(NTT4x30Avx512Rayon);
    impl_ckks_mul_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_neg_reference!(NTT4x30Avx512Rayon);
    impl_ckks_neg_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_pow2_reference!(NTT4x30Avx512Rayon);
    impl_ckks_pow2_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_rotate_reference!(NTT4x30Avx512Rayon);
    impl_ckks_rotate_reference!(crate::NTT4x30CIAvx512Rayon);
    select_avx512_encoding_transform!(NTT4x30Avx512Rayon);
    select_avx512_encoding_transform!(crate::NTT4x30CIAvx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30Avx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(crate::NTT4x30CIAvx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30Avx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::NTT4x30CIAvx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30Avx512Rayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_add_reference!(NTT4x30Avx512Rayon);
    impl_ckks_add_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_sub_reference!(NTT4x30Avx512Rayon);
    impl_ckks_sub_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_plaintext_reference!(NTT4x30Avx512Rayon);
    impl_ckks_plaintext_reference!(crate::NTT4x30CIAvx512Rayon);
    impl_ckks_dft_reference!(NTT4x30Avx512Rayon);
    impl_ckks_dft_reference!(crate::NTT4x30CIAvx512Rayon);
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
mod ifma_rayon_defaults {
    use super::*;

    impl_ckks_conjugate_reference!(NTT3x42IfmaRayon);
    impl_ckks_conjugate_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_copy_reference!(NTT3x42IfmaRayon);
    impl_ckks_copy_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_encryption_reference!(NTT3x42IfmaRayon);
    impl_ckks_encryption_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_imag_reference!(NTT3x42IfmaRayon);
    impl_ckks_imag_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_mul_reference!(NTT3x42IfmaRayon);
    impl_ckks_mul_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_neg_reference!(NTT3x42IfmaRayon);
    impl_ckks_neg_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_pow2_reference!(NTT3x42IfmaRayon);
    impl_ckks_pow2_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_rotate_reference!(NTT3x42IfmaRayon);
    impl_ckks_rotate_reference!(crate::NTT3x42CIIfmaRayon);
    select_avx512_encoding_transform!(NTT3x42IfmaRayon);
    select_avx512_encoding_transform!(crate::NTT3x42CIIfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(NTT3x42IfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(crate::NTT3x42CIIfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT3x42IfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::NTT3x42CIIfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT3x42IfmaRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_add_reference!(NTT3x42IfmaRayon);
    impl_ckks_add_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_sub_reference!(NTT3x42IfmaRayon);
    impl_ckks_sub_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_plaintext_reference!(NTT3x42IfmaRayon);
    impl_ckks_plaintext_reference!(crate::NTT3x42CIIfmaRayon);
    impl_ckks_dft_reference!(NTT3x42IfmaRayon);
    impl_ckks_dft_reference!(crate::NTT3x42CIIfmaRayon);
}
