#[cfg(any(test, feature = "enable-test-suite"))]
mod comparison {
    use crate::test_suite::ControlledSamplingFFT64Ref;
    poulpy_ckks::impl_ckks_plaintext_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_copy_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_add_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_sub_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_neg_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_pow2_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_imag_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_rotate_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_conjugate_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_encryption_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_mul_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_polynomial_evaluation_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_eval_mod_reference!(ControlledSamplingFFT64Ref);
}
