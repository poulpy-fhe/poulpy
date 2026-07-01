use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_avx512_f64,
    backend = crate::FFT64Avx512,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_avx512_f64,
    backend = crate::NTT4x30Avx512,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_avx512_f128,
    backend = crate::NTT4x30Avx512,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt3x42_ifma_f64,
    backend = crate::NTT3x42Ifma,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt3x42_ifma_f128,
    backend = crate::NTT3x42Ifma,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
