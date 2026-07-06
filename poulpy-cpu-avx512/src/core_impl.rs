#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_keyswitch_defaults_full,
    impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full,
    impl_lwe_keyswitch_defaults_full,
};

impl_glwe_automorphism_defaults_full!(FFT64Avx512);
impl_glwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_ggsw_automorphism_defaults_full!(FFT64Avx512);
impl_ggsw_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_automorphism_defaults_full!(NTT3x42Ifma);

impl_gglwe_automorphism_defaults_full!(FFT64Avx512);
impl_gglwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_decryption_defaults_full!(FFT64Avx512);
impl_decryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_decryption_defaults_full!(NTT3x42Ifma);

impl_glwe_trace_defaults_full!(FFT64Avx512);
impl_glwe_trace_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_trace_defaults_full!(NTT3x42Ifma);

impl_glwe_packing_defaults_full!(FFT64Avx512);
impl_glwe_packing_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_packing_defaults_full!(NTT3x42Ifma);

impl_conversion_defaults_full!(FFT64Avx512);
impl_conversion_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_conversion_defaults_full!(NTT3x42Ifma);

impl_glwe_keyswitch_defaults_full!(FFT64Avx512);
impl_glwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_gglwe_keyswitch_defaults_full!(FFT64Avx512);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_ggsw_keyswitch_defaults_full!(FFT64Avx512);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_keyswitch_defaults_full!(NTT3x42Ifma);

impl_lwe_keyswitch_defaults_full!(FFT64Avx512);
impl_lwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_lwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_encryption_defaults_full!(FFT64Avx512);
impl_encryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_encryption_defaults_full!(NTT3x42Ifma);

impl_glwe_external_product_defaults_full!(FFT64Avx512);
impl_glwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_external_product_defaults_full!(NTT3x42Ifma);

impl_gglwe_external_product_defaults_full!(FFT64Avx512);
impl_gglwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_external_product_defaults_full!(NTT3x42Ifma);

impl_ggsw_external_product_defaults_full!(FFT64Avx512);
impl_ggsw_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_external_product_defaults_full!(NTT3x42Ifma);

impl_linear_transformation_defaults_full!(FFT64Avx512);
impl_linear_transformation_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_linear_transformation_defaults_full!(NTT3x42Ifma);
