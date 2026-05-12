use crate::{FFT64Avx, NTT120Avx};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_keyswitch_defaults_full,
    impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full, impl_lwe_keyswitch_defaults_full,
};

impl_glwe_automorphism_defaults_full!(FFT64Avx);
impl_glwe_automorphism_defaults_full!(NTT120Avx);

impl_ggsw_automorphism_defaults_full!(FFT64Avx);
impl_ggsw_automorphism_defaults_full!(NTT120Avx);
impl_gglwe_automorphism_defaults_full!(FFT64Avx);
impl_gglwe_automorphism_defaults_full!(NTT120Avx);

impl_decryption_defaults_full!(FFT64Avx);
impl_decryption_defaults_full!(NTT120Avx);
impl_glwe_trace_defaults_full!(FFT64Avx);
impl_glwe_trace_defaults_full!(NTT120Avx);
impl_glwe_packing_defaults_full!(FFT64Avx);
impl_glwe_packing_defaults_full!(NTT120Avx);

impl_conversion_defaults_full!(FFT64Avx);
impl_conversion_defaults_full!(NTT120Avx);

impl_glwe_keyswitch_defaults_full!(FFT64Avx);
impl_glwe_keyswitch_defaults_full!(NTT120Avx);
impl_gglwe_keyswitch_defaults_full!(FFT64Avx);
impl_gglwe_keyswitch_defaults_full!(NTT120Avx);
impl_ggsw_keyswitch_defaults_full!(FFT64Avx);
impl_ggsw_keyswitch_defaults_full!(NTT120Avx);
impl_lwe_keyswitch_defaults_full!(FFT64Avx);
impl_lwe_keyswitch_defaults_full!(NTT120Avx);

impl_encryption_defaults_full!(FFT64Avx);
impl_encryption_defaults_full!(NTT120Avx);

impl_glwe_external_product_defaults_full!(FFT64Avx);
impl_glwe_external_product_defaults_full!(NTT120Avx);
impl_gglwe_external_product_defaults_full!(FFT64Avx);
impl_gglwe_external_product_defaults_full!(NTT120Avx);
impl_ggsw_external_product_defaults_full!(FFT64Avx);
impl_ggsw_external_product_defaults_full!(NTT120Avx);
