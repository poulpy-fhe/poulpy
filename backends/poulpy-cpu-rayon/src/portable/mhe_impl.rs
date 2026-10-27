//! Reference multiparty implementations of the portable Rayon backends, each
//! paired with the parity suite against its serial base.

poulpy_mhe::impl_mhe_reference_full!(crate::FFT64PortableRayon);
poulpy_mhe::impl_mhe_reference_full!(crate::NTT4x30PortableRayon);

#[cfg(test)]
poulpy_mhe::mhe_parity_test_suite!(
    mod mhe_parity_fft64portablerayon,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64PortableRayon,
);
#[cfg(test)]
poulpy_mhe::mhe_parity_test_suite!(
    mod mhe_parity_ntt4x30portablerayon,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30PortableRayon,
);
