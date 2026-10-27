//! Reference multiparty implementations, each paired with the backend test
//! suite and the parity suite against the controlled-sampling FFT64 oracle.

poulpy_mhe::impl_mhe_reference_full!(crate::FFT64Portable);
poulpy_mhe::impl_mhe_reference_full!(crate::NTT4x30Portable);

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64portable, backend = crate::FFT64Portable);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30portable, backend = crate::NTT4x30Portable);
#[cfg(test)]
poulpy_mhe::mhe_parity_test_suite!(
    mod mhe_parity_fft64portable,
    backend_ref = poulpy_cpu_oracle::test_suite::ControlledSamplingFFT64Oracle,
    backend_test = crate::FFT64Portable,
);
#[cfg(test)]
poulpy_mhe::mhe_parity_test_suite!(
    mod mhe_parity_ntt4x30portable,
    backend_ref = poulpy_cpu_oracle::test_suite::ControlledSamplingFFT64Oracle,
    backend_test = crate::NTT4x30Portable,
);
