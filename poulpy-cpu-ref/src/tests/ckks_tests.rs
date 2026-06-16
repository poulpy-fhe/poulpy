//! CKKS conformance tests for the reference backends.
//!
//! All test logic lives in `poulpy_ckks::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `ckks_backend_test_suite!`.
//! There should be no hand-written test here.

use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f64,
    backend = crate::NTT120Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f128,
    backend = crate::NTT120Ref,
    scalar = f128::f128,
    encoder = crate::FFT64ReimTable<f128::f128>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
