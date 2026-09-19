use crate::{FFT64Ref, NTT4x30Ref};
use poulpy_core::{
    impl_conversion_reference_full, impl_decryption_reference_full, impl_encryption_reference_full,
    impl_gglwe_automorphism_reference_full, impl_gglwe_product_digits_strided_reference, impl_glwe_automorphism_reference_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_packing_reference_full,
    impl_glwe_tensoring_reference, impl_glwe_trace_reference_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
};

impl_glwe_tensoring_reference!(FFT64Ref);
impl_glwe_tensoring_reference!(NTT4x30Ref);
impl_gglwe_product_digits_strided_reference!(FFT64Ref);
impl_gglwe_product_digits_strided_reference!(NTT4x30Ref);

impl_glwe_automorphism_reference_full!(FFT64Ref);
impl_glwe_automorphism_reference_full!(NTT4x30Ref);

impl_gglwe_automorphism_reference_full!(FFT64Ref);
impl_gglwe_automorphism_reference_full!(NTT4x30Ref);

impl_decryption_reference_full!(FFT64Ref);
impl_decryption_reference_full!(NTT4x30Ref);
impl_glwe_trace_reference_full!(FFT64Ref);
impl_glwe_trace_reference_full!(NTT4x30Ref);
impl_glwe_packing_reference_full!(FFT64Ref);
impl_glwe_packing_reference_full!(NTT4x30Ref);

impl_conversion_reference_full!(FFT64Ref);
impl_conversion_reference_full!(NTT4x30Ref);

impl_glwe_keyswitch_reference_full!(FFT64Ref);
impl_glwe_keyswitch_reference_full!(NTT4x30Ref);
impl_lwe_keyswitch_reference_full!(FFT64Ref);
impl_lwe_keyswitch_reference_full!(NTT4x30Ref);

impl_encryption_reference_full!(FFT64Ref);
crate::impl_sampling_host!(FFT64Ref, fft64);
impl_encryption_reference_full!(NTT4x30Ref);
crate::impl_sampling_host!(NTT4x30Ref, ntt4x30);

impl_glwe_external_product_reference_full!(FFT64Ref);
impl_glwe_external_product_reference_full!(NTT4x30Ref);

impl_linear_transformation_reference_full!(FFT64Ref);
impl_linear_transformation_reference_full!(NTT4x30Ref);
