//! Reference multiparty implementations, each paired with the backend test suite.

poulpy_mhe::impl_mhe_reference_full!(crate::FFT64Ref);
poulpy_mhe::impl_mhe_reference_full!(crate::NTT4x30Ref);

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64ref, backend = crate::FFT64Ref);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30ref, backend = crate::NTT4x30Ref);
