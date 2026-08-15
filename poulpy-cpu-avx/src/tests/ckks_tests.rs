use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Avx,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_rayon_f64,
    backend = crate::NTT4x30AvxRayon,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Avx,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);
