//! Binary-FHE implementations of the portable Rayon backends, each paired with the complete conformance suite.

macro_rules! register_backend {
    ($backend:ty, $comparison:ty, $suite:ident) => {
        poulpy_bin_fhe::impl_bin_fhe_reference_full!($backend, scheduling = parallel);
        #[cfg(test)]
        mod $suite {
            poulpy_bin_fhe::bin_fhe_parity_test_suite!(mod paired, backend_ref = $comparison, backend_test = $backend);
            poulpy_bin_fhe::bin_fhe_reference_test_suite!(mod reference, backend = $backend);
        }
    };
}

register_backend!(
    crate::FFT64PortableRayon,
    poulpy_cpu_portable::FFT64Portable,
    bin_fhe_parity_fft64portablerayon
);
register_backend!(
    crate::NTT4x30PortableRayon,
    poulpy_cpu_portable::NTT4x30Portable,
    bin_fhe_parity_ntt4x30portablerayon
);
