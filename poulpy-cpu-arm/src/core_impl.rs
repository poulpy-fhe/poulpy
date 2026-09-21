use crate::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{
    impl_automorphism_reference_full, impl_conversion_reference_full, impl_decryption_reference_full,
    impl_encryption_reference_full, impl_gglwe_external_product_derived_full, impl_gglwe_keyswitch_derived_full,
    impl_gglwe_product_digits_strided_reference, impl_ggsw_external_product_derived_full, impl_ggsw_keyswitch_derived_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_packing_derived_full,
    impl_glwe_tensoring_reference, impl_glwe_trace_derived_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_gglwe_product_digits_strided_reference!(NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30NeonRayon);

impl_automorphism_reference_full!(FFT64Neon);
impl_automorphism_reference_full!(NTT4x30Neon);

impl_decryption_reference_full!(FFT64Neon);
impl_decryption_reference_full!(NTT4x30Neon);
impl_glwe_trace_derived_full!(FFT64Neon);
impl_glwe_trace_derived_full!(NTT4x30Neon);
impl_glwe_packing_derived_full!(FFT64Neon);
impl_glwe_packing_derived_full!(NTT4x30Neon);

impl_conversion_reference_full!(FFT64Neon);
impl_conversion_reference_full!(NTT4x30Neon);

impl_glwe_keyswitch_reference_full!(FFT64Neon);
impl_glwe_keyswitch_reference_full!(NTT4x30Neon);
impl_gglwe_keyswitch_derived_full!(FFT64Neon);
impl_gglwe_keyswitch_derived_full!(NTT4x30Neon);
impl_ggsw_keyswitch_derived_full!(FFT64Neon);
impl_ggsw_keyswitch_derived_full!(NTT4x30Neon);
impl_lwe_keyswitch_reference_full!(FFT64Neon);
impl_lwe_keyswitch_reference_full!(NTT4x30Neon);

impl_encryption_reference_full!(FFT64Neon);
poulpy_core::impl_operations_reference_full!(FFT64Neon);
poulpy_core::impl_polynomial_evaluation_derived_full!(FFT64Neon);
poulpy_cpu_ref::impl_sampling_host!(FFT64Neon, fft64);
impl_encryption_reference_full!(NTT4x30Neon);
poulpy_core::impl_operations_reference_full!(NTT4x30Neon);
poulpy_core::impl_polynomial_evaluation_derived_full!(NTT4x30Neon);
poulpy_cpu_ref::impl_sampling_host!(NTT4x30Neon, ntt4x30);

impl_glwe_external_product_reference_full!(FFT64Neon);
impl_glwe_external_product_reference_full!(NTT4x30Neon);
impl_gglwe_external_product_derived_full!(FFT64Neon);
impl_gglwe_external_product_derived_full!(NTT4x30Neon);
impl_ggsw_external_product_derived_full!(FFT64Neon);
impl_ggsw_external_product_derived_full!(NTT4x30Neon);

impl_linear_transformation_reference_full!(FFT64Neon);
impl_linear_transformation_reference_full!(NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    macro_rules! impl_core_defaults {
        ($backend:ty) => {
            impl_automorphism_reference_full!($backend);
            impl_decryption_reference_full!($backend);
            impl_glwe_trace_derived_full!($backend);
            impl_glwe_packing_derived_full!($backend);
            impl_conversion_reference_full!($backend);
            impl_glwe_keyswitch_reference_full!($backend);
            impl_gglwe_keyswitch_derived_full!($backend);
            impl_ggsw_keyswitch_derived_full!($backend);
            impl_lwe_keyswitch_reference_full!($backend);
            impl_encryption_reference_full!($backend);
            poulpy_core::impl_operations_reference_full!($backend);
            poulpy_core::impl_polynomial_evaluation_derived_full!($backend);
            impl_glwe_external_product_reference_full!($backend);
            impl_gglwe_external_product_derived_full!($backend);
            impl_ggsw_external_product_derived_full!($backend);
            impl_linear_transformation_reference_full!($backend);
        };
    }

    impl_core_defaults!(FFT64NeonRayon);
    impl_core_defaults!(NTT4x30NeonRayon);

    poulpy_cpu_ref::impl_sampling_host!(FFT64NeonRayon, fft64);
    poulpy_cpu_ref::impl_sampling_host!(NTT4x30NeonRayon, ntt4x30);
}
