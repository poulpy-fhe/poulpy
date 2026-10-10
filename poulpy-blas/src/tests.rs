//! Suite instantiations; no hand-written test here.

crate::impl_blas_reference_full!(poulpy_cpu_oracle::FFT64Oracle);
crate::impl_blas_reference_full!(poulpy_cpu_oracle::NTT4x30Oracle);

crate::blas_backend_test_suite!(
    mod fft64_oracle,
    backend = poulpy_cpu_oracle::FFT64Oracle,
    scalar = f64,
    encoder = poulpy_cpu_oracle::EncodingFft<f64>,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

crate::blas_backend_test_suite!(
    mod ntt4x30_oracle,
    backend = poulpy_cpu_oracle::NTT4x30Oracle,
    scalar = f64,
    encoder = poulpy_cpu_oracle::EncodingFft<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

poulpy_ckks::ckks_parity_test_suite! {
    mod blas_parity_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = poulpy_cpu_portable::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        inter => crate::test_suite::parity::test_blas_inter_parity,
        intra => crate::test_suite::parity::test_blas_intra_parity,
        matmul => crate::test_suite::parity::test_blas_matmul_parity,
    },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod blas_parity_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = poulpy_cpu_portable::NTT4x30Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        inter => crate::test_suite::parity::test_blas_inter_parity,
        intra => crate::test_suite::parity::test_blas_intra_parity,
        matmul => crate::test_suite::parity::test_blas_matmul_parity,
    },
}
