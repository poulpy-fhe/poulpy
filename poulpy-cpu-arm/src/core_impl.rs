use crate::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{
    impl_conversion_reference_full, impl_decryption_reference_full, impl_encryption_reference_full,
    impl_gglwe_automorphism_reference_full, impl_gglwe_external_product_reference_full, impl_gglwe_keyswitch_reference_full,
    impl_gglwe_product_digits_strided_reference, impl_ggsw_automorphism_reference_full,
    impl_ggsw_external_product_reference_full, impl_ggsw_keyswitch_reference_full, impl_glwe_automorphism_reference_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_packing_reference_full,
    impl_glwe_tensoring_reference, impl_glwe_trace_reference_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(crate::FFT64CINeon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_glwe_tensoring_reference!(crate::NTT4x30CINeon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_gglwe_product_digits_strided_reference!(crate::FFT64CINeon);
impl_gglwe_product_digits_strided_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(crate::NTT4x30CINeon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(crate::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(crate::NTT4x30CINeonRayon);

impl_glwe_automorphism_reference_full!(FFT64Neon);
impl_glwe_automorphism_reference_full!(crate::FFT64CINeon);
impl_glwe_automorphism_reference_full!(NTT4x30Neon);
impl_glwe_automorphism_reference_full!(crate::NTT4x30CINeon);

impl_ggsw_automorphism_reference_full!(FFT64Neon);
impl_ggsw_automorphism_reference_full!(crate::FFT64CINeon);
impl_ggsw_automorphism_reference_full!(NTT4x30Neon);
impl_ggsw_automorphism_reference_full!(crate::NTT4x30CINeon);
impl_gglwe_automorphism_reference_full!(FFT64Neon);
impl_gglwe_automorphism_reference_full!(crate::FFT64CINeon);
impl_gglwe_automorphism_reference_full!(NTT4x30Neon);
impl_gglwe_automorphism_reference_full!(crate::NTT4x30CINeon);

impl_decryption_reference_full!(FFT64Neon);
impl_decryption_reference_full!(crate::FFT64CINeon);
impl_decryption_reference_full!(NTT4x30Neon);
impl_decryption_reference_full!(crate::NTT4x30CINeon);
impl_glwe_trace_reference_full!(FFT64Neon);
impl_glwe_trace_reference_full!(crate::FFT64CINeon);
impl_glwe_trace_reference_full!(NTT4x30Neon);
impl_glwe_trace_reference_full!(crate::NTT4x30CINeon);
impl_glwe_packing_reference_full!(FFT64Neon);
impl_glwe_packing_reference_full!(crate::FFT64CINeon);
impl_glwe_packing_reference_full!(NTT4x30Neon);
impl_glwe_packing_reference_full!(crate::NTT4x30CINeon);

impl_conversion_reference_full!(FFT64Neon);
impl_conversion_reference_full!(crate::FFT64CINeon);
impl_conversion_reference_full!(NTT4x30Neon);
impl_conversion_reference_full!(crate::NTT4x30CINeon);

impl_glwe_keyswitch_reference_full!(FFT64Neon);
impl_glwe_keyswitch_reference_full!(crate::FFT64CINeon);
impl_glwe_keyswitch_reference_full!(NTT4x30Neon);
impl_glwe_keyswitch_reference_full!(crate::NTT4x30CINeon);
impl_gglwe_keyswitch_reference_full!(FFT64Neon);
impl_gglwe_keyswitch_reference_full!(crate::FFT64CINeon);
impl_gglwe_keyswitch_reference_full!(NTT4x30Neon);
impl_gglwe_keyswitch_reference_full!(crate::NTT4x30CINeon);
impl_ggsw_keyswitch_reference_full!(FFT64Neon);
impl_ggsw_keyswitch_reference_full!(crate::FFT64CINeon);
impl_ggsw_keyswitch_reference_full!(NTT4x30Neon);
impl_ggsw_keyswitch_reference_full!(crate::NTT4x30CINeon);
impl_lwe_keyswitch_reference_full!(FFT64Neon);
impl_lwe_keyswitch_reference_full!(crate::FFT64CINeon);
impl_lwe_keyswitch_reference_full!(NTT4x30Neon);
impl_lwe_keyswitch_reference_full!(crate::NTT4x30CINeon);

impl_encryption_reference_full!(FFT64Neon);
impl_encryption_reference_full!(crate::FFT64CINeon);
poulpy_cpu_ref::impl_sampling_host!(FFT64Neon, fft64);
poulpy_cpu_ref::impl_sampling_host!(crate::FFT64CINeon, fft64);
impl_encryption_reference_full!(NTT4x30Neon);
impl_encryption_reference_full!(crate::NTT4x30CINeon);
poulpy_cpu_ref::impl_sampling_host!(NTT4x30Neon, ntt4x30);
poulpy_cpu_ref::impl_sampling_host!(crate::NTT4x30CINeon, ntt4x30);

impl_glwe_external_product_reference_full!(FFT64Neon);
impl_glwe_external_product_reference_full!(crate::FFT64CINeon);
impl_glwe_external_product_reference_full!(NTT4x30Neon);
impl_glwe_external_product_reference_full!(crate::NTT4x30CINeon);
impl_gglwe_external_product_reference_full!(FFT64Neon);
impl_gglwe_external_product_reference_full!(crate::FFT64CINeon);
impl_gglwe_external_product_reference_full!(NTT4x30Neon);
impl_gglwe_external_product_reference_full!(crate::NTT4x30CINeon);
impl_ggsw_external_product_reference_full!(FFT64Neon);
impl_ggsw_external_product_reference_full!(crate::FFT64CINeon);
impl_ggsw_external_product_reference_full!(NTT4x30Neon);
impl_ggsw_external_product_reference_full!(crate::NTT4x30CINeon);

impl_linear_transformation_reference_full!(FFT64Neon);
impl_linear_transformation_reference_full!(crate::FFT64CINeon);
impl_linear_transformation_reference_full!(NTT4x30Neon);
impl_linear_transformation_reference_full!(crate::NTT4x30CINeon);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    macro_rules! impl_core_defaults {
        ($backend:ty) => {
            impl_glwe_automorphism_reference_full!($backend);
            impl_ggsw_automorphism_reference_full!($backend);
            impl_gglwe_automorphism_reference_full!($backend);
            impl_decryption_reference_full!($backend);
            impl_glwe_trace_reference_full!($backend);
            impl_glwe_packing_reference_full!($backend);
            impl_conversion_reference_full!($backend);
            impl_glwe_keyswitch_reference_full!($backend);
            impl_gglwe_keyswitch_reference_full!($backend);
            impl_ggsw_keyswitch_reference_full!($backend);
            impl_lwe_keyswitch_reference_full!($backend);
            impl_encryption_reference_full!($backend);
            impl_glwe_external_product_reference_full!($backend);
            impl_gglwe_external_product_reference_full!($backend);
            impl_ggsw_external_product_reference_full!($backend);
            impl_linear_transformation_reference_full!($backend);
        };
    }

    impl_core_defaults!(FFT64NeonRayon);
    impl_core_defaults!(crate::FFT64CINeonRayon);
    impl_core_defaults!(NTT4x30NeonRayon);
    impl_core_defaults!(crate::NTT4x30CINeonRayon);

    poulpy_cpu_ref::impl_sampling_host!(FFT64NeonRayon, fft64);
    poulpy_cpu_ref::impl_sampling_host!(crate::FFT64CINeonRayon, fft64);
    poulpy_cpu_ref::impl_sampling_host!(NTT4x30NeonRayon, ntt4x30);
    poulpy_cpu_ref::impl_sampling_host!(crate::NTT4x30CINeonRayon, ntt4x30);
}
