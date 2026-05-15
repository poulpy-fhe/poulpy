use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Neon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f64,
    backend = crate::NTT120Neon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);
