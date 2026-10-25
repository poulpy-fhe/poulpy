//! Reference multiparty implementations, each paired with the backend test suite.

poulpy_mhe::impl_mhe_reference_full!(crate::FFT64Portable);
poulpy_mhe::impl_mhe_reference_full!(crate::NTT4x30Portable);

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64portable, backend = crate::FFT64Portable);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30portable, backend = crate::NTT4x30Portable);
