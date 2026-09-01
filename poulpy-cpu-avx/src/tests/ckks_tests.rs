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
    mod fft64_rayon_f64,
    backend = crate::FFT64AvxRayon,
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

/// Full logN16 bootstraps per preset: slow, so opt in with `--ignored`.
mod bootstrapping_presets {
    use poulpy_ckks::test_suite::presets::bootstrapping_presets_meet_precision;

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn ntt4x30_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::NTT4x30Avx>();
    }

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn fft64_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::FFT64Avx>();
    }
}
