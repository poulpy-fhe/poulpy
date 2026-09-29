//! The multiparty backend test suite on both reference backends.

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64ref, backend = crate::FFT64Ref);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30ref, backend = crate::NTT4x30Ref);
