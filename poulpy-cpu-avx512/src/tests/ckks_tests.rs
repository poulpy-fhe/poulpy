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
    mod ntt120_avx512_f64,
    backend = crate::NTT120Avx512,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_avx512_f128,
    backend = crate::NTT120Avx512,
    scalar = f128::f128,
    encoder = poulpy_cpu_ref::FFT64ReimTable<f128::f128>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt126_ifma_f64,
    backend = crate::NTT126Ifma,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt126_ifma_f128,
    backend = crate::NTT126Ifma,
    scalar = f128::f128,
    encoder = poulpy_cpu_ref::FFT64ReimTable<f128::f128>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
