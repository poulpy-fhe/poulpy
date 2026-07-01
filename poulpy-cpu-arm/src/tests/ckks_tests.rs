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
    mod ntt4x30_f64,
    backend = crate::NTT4x30Neon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

// binary128 path on aarch64: `Quad` scalar with the reference encoder.
ckks_backend_test_suite!(
    mod ntt4x30_f128,
    backend = crate::NTT4x30Neon,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
