use crate::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_ckks::{
    impl_ckks_add_reference, impl_ckks_conjugate_reference, impl_ckks_copy_reference, impl_ckks_dft_reference,
    impl_ckks_encapsulated_mod_up_reference, impl_ckks_encryption_reference, impl_ckks_imag_reference, impl_ckks_mul_reference,
    impl_ckks_neg_reference, impl_ckks_plaintext_reference, impl_ckks_pow2_reference, impl_ckks_rotate_reference,
    impl_ckks_sub_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Neon);
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CINeon);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Neon);
impl_ckks_encapsulated_mod_up_reference!(crate::NTT4x30CINeon);
impl_ckks_conjugate_reference!(FFT64Neon);
impl_ckks_conjugate_reference!(crate::FFT64CINeon);
impl_ckks_conjugate_reference!(NTT4x30Neon);
impl_ckks_conjugate_reference!(crate::NTT4x30CINeon);
impl_ckks_copy_reference!(FFT64Neon);
impl_ckks_copy_reference!(crate::FFT64CINeon);
impl_ckks_copy_reference!(NTT4x30Neon);
impl_ckks_copy_reference!(crate::NTT4x30CINeon);
impl_ckks_encryption_reference!(FFT64Neon);
impl_ckks_encryption_reference!(crate::FFT64CINeon);
impl_ckks_encryption_reference!(NTT4x30Neon);
impl_ckks_encryption_reference!(crate::NTT4x30CINeon);
impl_ckks_imag_reference!(FFT64Neon);
impl_ckks_imag_reference!(crate::FFT64CINeon);
impl_ckks_imag_reference!(NTT4x30Neon);
impl_ckks_imag_reference!(crate::NTT4x30CINeon);
impl_ckks_mul_reference!(FFT64Neon);
impl_ckks_mul_reference!(crate::FFT64CINeon);
impl_ckks_mul_reference!(NTT4x30Neon);
impl_ckks_mul_reference!(crate::NTT4x30CINeon);
impl_ckks_neg_reference!(FFT64Neon);
impl_ckks_neg_reference!(crate::FFT64CINeon);
impl_ckks_neg_reference!(NTT4x30Neon);
impl_ckks_neg_reference!(crate::NTT4x30CINeon);
impl_ckks_pow2_reference!(FFT64Neon);
impl_ckks_pow2_reference!(crate::FFT64CINeon);
impl_ckks_pow2_reference!(NTT4x30Neon);
impl_ckks_pow2_reference!(crate::NTT4x30CINeon);
impl_ckks_rotate_reference!(FFT64Neon);
impl_ckks_rotate_reference!(crate::FFT64CINeon);
impl_ckks_rotate_reference!(NTT4x30Neon);
impl_ckks_rotate_reference!(crate::NTT4x30CINeon);
// `f64` encodes through the NEON kernels; `Quad` has no accelerated transform
// and falls back to the generic scalar table. Rust has no specialization, so
// accelerated backends list their precisions explicitly.
macro_rules! select_neon_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = crate::FFT64NeonReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_neon_encoding_transform!(FFT64Neon);
select_neon_encoding_transform!(crate::FFT64CINeon);
select_neon_encoding_transform!(NTT4x30Neon);
select_neon_encoding_transform!(crate::NTT4x30CINeon);

::poulpy_cpu_ref::impl_ckks_encoding!(FFT64Neon);
::poulpy_cpu_ref::impl_ckks_encoding!(crate::FFT64CINeon);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64Neon);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::FFT64CINeon);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64Neon);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::FFT64CINeon);
::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30Neon);
::poulpy_cpu_ref::impl_ckks_encoding!(crate::NTT4x30CINeon);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30Neon);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(crate::NTT4x30CINeon);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30Neon);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(crate::NTT4x30CINeon);
impl_ckks_add_reference!(FFT64Neon);
impl_ckks_add_reference!(crate::FFT64CINeon);
impl_ckks_add_reference!(NTT4x30Neon);
impl_ckks_add_reference!(crate::NTT4x30CINeon);
impl_ckks_sub_reference!(FFT64Neon);
impl_ckks_sub_reference!(crate::FFT64CINeon);
impl_ckks_sub_reference!(NTT4x30Neon);
impl_ckks_sub_reference!(crate::NTT4x30CINeon);
impl_ckks_plaintext_reference!(FFT64Neon);
impl_ckks_plaintext_reference!(crate::FFT64CINeon);
impl_ckks_plaintext_reference!(NTT4x30Neon);
impl_ckks_plaintext_reference!(crate::NTT4x30CINeon);
impl_ckks_dft_reference!(FFT64Neon);
impl_ckks_dft_reference!(crate::FFT64CINeon);
impl_ckks_dft_reference!(NTT4x30Neon);
impl_ckks_dft_reference!(crate::NTT4x30CINeon);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    macro_rules! impl_ckks_defaults {
        ($backend:ty) => {
            impl_ckks_encapsulated_mod_up_reference!($backend);
            impl_ckks_conjugate_reference!($backend);
            impl_ckks_copy_reference!($backend);
            impl_ckks_encryption_reference!($backend);
            impl_ckks_imag_reference!($backend);
            impl_ckks_mul_reference!($backend);
            impl_ckks_neg_reference!($backend);
            impl_ckks_pow2_reference!($backend);
            impl_ckks_rotate_reference!($backend);
            select_neon_encoding_transform!($backend);
            ::poulpy_cpu_ref::impl_ckks_encoding!($backend);
            ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!($backend);
            ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!($backend);
            impl_ckks_add_reference!($backend);
            impl_ckks_sub_reference!($backend);
            impl_ckks_plaintext_reference!($backend);
            impl_ckks_dft_reference!($backend);
        };
    }

    impl_ckks_defaults!(FFT64NeonRayon);
    impl_ckks_defaults!(crate::FFT64CINeonRayon);
    impl_ckks_defaults!(NTT4x30NeonRayon);
    impl_ckks_defaults!(crate::NTT4x30CINeonRayon);
}
