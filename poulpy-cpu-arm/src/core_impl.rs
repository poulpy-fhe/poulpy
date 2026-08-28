use crate::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensoring_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};

impl_glwe_tensoring_default!(FFT64Neon);
impl_glwe_tensoring_default!(NTT4x30Neon);
impl_gglwe_product_digits_strided_default!(FFT64Neon);
impl_gglwe_product_digits_strided_default!(NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_default!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_default!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_default!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_default!(NTT4x30NeonRayon);

impl_glwe_automorphism_defaults_full!(FFT64Neon);
impl_glwe_automorphism_defaults_full!(NTT4x30Neon);

impl_ggsw_automorphism_defaults_full!(FFT64Neon);
impl_ggsw_automorphism_defaults_full!(NTT4x30Neon);
impl_gglwe_automorphism_defaults_full!(FFT64Neon);
impl_gglwe_automorphism_defaults_full!(NTT4x30Neon);

impl_decryption_defaults_full!(FFT64Neon);
impl_decryption_defaults_full!(NTT4x30Neon);
impl_glwe_trace_defaults_full!(FFT64Neon);
impl_glwe_trace_defaults_full!(NTT4x30Neon);
impl_glwe_packing_defaults_full!(FFT64Neon);
impl_glwe_packing_defaults_full!(NTT4x30Neon);

impl_conversion_defaults_full!(FFT64Neon);
impl_conversion_defaults_full!(NTT4x30Neon);

impl_glwe_keyswitch_defaults_full!(FFT64Neon);
impl_glwe_keyswitch_defaults_full!(NTT4x30Neon);
impl_gglwe_keyswitch_defaults_full!(FFT64Neon);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Neon);
impl_ggsw_keyswitch_defaults_full!(FFT64Neon);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Neon);
impl_lwe_keyswitch_defaults_full!(FFT64Neon);
impl_lwe_keyswitch_defaults_full!(NTT4x30Neon);

impl_encryption_defaults_full!(FFT64Neon);
impl_encryption_defaults_full!(NTT4x30Neon);

impl_glwe_external_product_defaults_full!(FFT64Neon);
impl_glwe_external_product_defaults_full!(NTT4x30Neon);
impl_gglwe_external_product_defaults_full!(FFT64Neon);
impl_gglwe_external_product_defaults_full!(NTT4x30Neon);
impl_ggsw_external_product_defaults_full!(FFT64Neon);
impl_ggsw_external_product_defaults_full!(NTT4x30Neon);

impl_linear_transformation_defaults_full!(FFT64Neon);
impl_linear_transformation_defaults_full!(NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    macro_rules! impl_core_defaults {
        ($backend:ty) => {
            impl_glwe_automorphism_defaults_full!($backend);
            impl_ggsw_automorphism_defaults_full!($backend);
            impl_gglwe_automorphism_defaults_full!($backend);
            impl_decryption_defaults_full!($backend);
            impl_glwe_trace_defaults_full!($backend);
            impl_glwe_packing_defaults_full!($backend);
            impl_conversion_defaults_full!($backend);
            impl_glwe_keyswitch_defaults_full!($backend);
            impl_gglwe_keyswitch_defaults_full!($backend);
            impl_ggsw_keyswitch_defaults_full!($backend);
            impl_lwe_keyswitch_defaults_full!($backend);
            impl_encryption_defaults_full!($backend);
            impl_glwe_external_product_defaults_full!($backend);
            impl_gglwe_external_product_defaults_full!($backend);
            impl_ggsw_external_product_defaults_full!($backend);
            impl_linear_transformation_defaults_full!($backend);
        };
    }

    impl_core_defaults!(FFT64NeonRayon);
    impl_core_defaults!(NTT4x30NeonRayon);
}
