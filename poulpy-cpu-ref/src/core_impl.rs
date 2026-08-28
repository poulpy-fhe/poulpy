use crate::{FFT64Ref, NTT4x30Ref};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_bound_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensoring_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};

impl_glwe_tensoring_default!(FFT64Ref);
impl_glwe_tensoring_default!(NTT4x30Ref);
impl_gglwe_product_bound_default!(FFT64Ref);
impl_gglwe_product_bound_default!(NTT4x30Ref);

impl_glwe_automorphism_defaults_full!(FFT64Ref);
impl_glwe_automorphism_defaults_full!(NTT4x30Ref);

impl_ggsw_automorphism_defaults_full!(FFT64Ref);
impl_ggsw_automorphism_defaults_full!(NTT4x30Ref);
impl_gglwe_automorphism_defaults_full!(FFT64Ref);
impl_gglwe_automorphism_defaults_full!(NTT4x30Ref);

impl_decryption_defaults_full!(FFT64Ref);
impl_decryption_defaults_full!(NTT4x30Ref);
impl_glwe_trace_defaults_full!(FFT64Ref);
impl_glwe_trace_defaults_full!(NTT4x30Ref);
impl_glwe_packing_defaults_full!(FFT64Ref);
impl_glwe_packing_defaults_full!(NTT4x30Ref);

impl_conversion_defaults_full!(FFT64Ref);
impl_conversion_defaults_full!(NTT4x30Ref);

impl_glwe_keyswitch_defaults_full!(FFT64Ref);
impl_glwe_keyswitch_defaults_full!(NTT4x30Ref);
impl_gglwe_keyswitch_defaults_full!(FFT64Ref);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Ref);
impl_ggsw_keyswitch_defaults_full!(FFT64Ref);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Ref);
impl_lwe_keyswitch_defaults_full!(FFT64Ref);
impl_lwe_keyswitch_defaults_full!(NTT4x30Ref);

impl_encryption_defaults_full!(FFT64Ref);
impl_encryption_defaults_full!(NTT4x30Ref);

impl_glwe_external_product_defaults_full!(FFT64Ref);
impl_glwe_external_product_defaults_full!(NTT4x30Ref);
impl_gglwe_external_product_defaults_full!(FFT64Ref);
impl_gglwe_external_product_defaults_full!(NTT4x30Ref);
impl_ggsw_external_product_defaults_full!(FFT64Ref);
impl_ggsw_external_product_defaults_full!(NTT4x30Ref);

impl_linear_transformation_defaults_full!(FFT64Ref);
impl_linear_transformation_defaults_full!(NTT4x30Ref);
