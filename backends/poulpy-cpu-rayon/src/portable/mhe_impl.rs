//! Reference multiparty implementations of the portable Rayon backends, each paired with the backend test suite.

poulpy_mhe::impl_mhe_reference_full!(crate::FFT64PortableRayon);
poulpy_mhe::impl_mhe_reference_full!(crate::NTT4x30PortableRayon);

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64portablerayon, backend = crate::FFT64PortableRayon);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30portablerayon, backend = crate::NTT4x30PortableRayon);
