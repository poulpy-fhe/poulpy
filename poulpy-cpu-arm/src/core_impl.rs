use crate::{FFT64Neon, NTT4x30Neon};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_finalize_big_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_keyswitch_into_big_defaults_full, impl_glwe_packing_defaults_full,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};

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
impl_glwe_keyswitch_into_big_defaults_full!(FFT64Neon);
impl_glwe_keyswitch_into_big_defaults_full!(NTT4x30Neon);
impl_glwe_finalize_big_defaults_full!(FFT64Neon);
impl_glwe_finalize_big_defaults_full!(NTT4x30Neon);
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
