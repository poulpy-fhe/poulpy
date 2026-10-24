#[cfg(any(test, feature = "enable-test-suite"))]
mod comparison {
    use crate::test_suite::ControlledSamplingFFT64Portable;
    poulpy_ckks::impl_ckks_plaintext_znx_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_copy_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_add_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_sub_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_neg_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_pow2_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_imag_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_rotate_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_conjugate_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_encryption_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_mul_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_dot_product_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_linear_transformation_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_polynomial_evaluation_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_eval_mod_reference_full!(ControlledSamplingFFT64Portable);

    use crate::test_suite::ControlledSamplingFFT64CIPortable;
    poulpy_ckks::impl_ckks_plaintext_znx_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_copy_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_add_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_sub_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_neg_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_pow2_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_rotate_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_encryption_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_mul_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_dot_product_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_linear_transformation_reference_full!(ControlledSamplingFFT64CIPortable);
    poulpy_ckks::impl_ckks_polynomial_evaluation_reference_full!(ControlledSamplingFFT64CIPortable);
}
