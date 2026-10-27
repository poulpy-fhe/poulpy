#[cfg(any(test, feature = "enable-test-suite"))]
mod comparison {
    use crate::test_suite::{ControlledSamplingFFT64CIPortable, ControlledSamplingFFT64Portable};

    poulpy_ckks::impl_ckks_reference_full!(ControlledSamplingFFT64Portable);
    // EvalMod's standard reference evaluates through multiplication by i and complex polynomial evaluation.
    poulpy_ckks::impl_ckks_imag_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_conjugate_reference_full!(ControlledSamplingFFT64Portable);
    poulpy_ckks::impl_ckks_complex_polynomial_evaluation_reference_full!(ControlledSamplingFFT64Portable);

    poulpy_ckks::impl_ckks_reference_full!(ControlledSamplingFFT64CIPortable);
}
