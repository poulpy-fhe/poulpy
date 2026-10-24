//! Explicit binary-FHE implementations, each paired with the complete conformance suite.

macro_rules! register_backend {
    ($backend:ty, $comparison:ty, $suite:ident $(, scheduling = $scheduling:ident)?) => {
        poulpy_bin_fhe::impl_bin_fhe_reference_full!($backend $(, scheduling = $scheduling)?);
        #[cfg(test)]
        mod $suite {
            poulpy_bin_fhe::bin_fhe_parity_test_suite!(mod paired, backend_ref = $comparison, backend_test = $backend);
            poulpy_bin_fhe::bin_fhe_reference_test_suite!(mod reference, backend = $backend);
        }
    };
}

register_backend!(
    crate::FFT64Portable,
    poulpy_cpu_oracle::FFT64Oracle,
    bin_fhe_parity_fft64portable
);
register_backend!(
    crate::NTT4x30Portable,
    poulpy_cpu_oracle::NTT4x30Oracle,
    bin_fhe_parity_ntt4x30portable
);
