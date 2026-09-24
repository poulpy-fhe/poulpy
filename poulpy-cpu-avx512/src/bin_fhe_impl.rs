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

register_backend!(crate::FFT64Avx512, poulpy_cpu_ref::FFT64Ref, bin_fhe_parity_fft64avx512);
register_backend!(crate::NTT4x30Avx512, poulpy_cpu_ref::NTT4x30Ref, bin_fhe_parity_ntt4x30avx512);
#[cfg(feature = "enable-ifma")]
register_backend!(crate::NTT3x42Ifma, poulpy_cpu_ref::NTT4x30Ref, bin_fhe_parity_ntt3x42ifma);
#[cfg(feature = "enable-rayon")]
register_backend!(
    crate::FFT64Avx512Rayon,
    crate::FFT64Avx512,
    bin_fhe_parity_fft64avx512rayon,
    scheduling = parallel
);
#[cfg(feature = "enable-rayon")]
register_backend!(
    crate::NTT4x30Avx512Rayon,
    crate::NTT4x30Avx512,
    bin_fhe_parity_ntt4x30avx512rayon,
    scheduling = parallel
);
#[cfg(all(feature = "enable-rayon", feature = "enable-ifma"))]
register_backend!(
    crate::NTT3x42IfmaRayon,
    crate::NTT3x42Ifma,
    bin_fhe_parity_ntt3x42ifmarayon,
    scheduling = parallel
);
