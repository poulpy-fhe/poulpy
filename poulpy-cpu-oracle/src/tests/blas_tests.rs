//! BLAS conformance tests for the oracle backends: the functional suites
//! against cleartext expectations, and parity between the FFT64 and NTT4x30
//! oracles.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `blas_backend_test_suite!`
//! and `blas_parity_test_suite!`. There should be no hand-written test here.

use poulpy_blas::{blas_backend_test_suite, blas_parity_test_suite};

blas_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Oracle,
    scalar = f64,
    encoder = crate::EncodingFft<f64>,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

blas_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Oracle,
    scalar = f64,
    encoder = crate::EncodingFft<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

// Parity between the two oracle families.
blas_parity_test_suite! {
    mod blas_parity_fft64oracle_f64,
    backend_ref = crate::NTT4x30Oracle,
    backend_test = crate::FFT64Oracle,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64oracle_quad,
    backend_ref = crate::NTT4x30Oracle,
    backend_test = crate::FFT64Oracle,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30oracle_f64,
    backend_ref = crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30oracle_quad,
    backend_ref = crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Conjugate-invariant backends run the same pairs.
blas_parity_test_suite! {
    mod blas_parity_ci_fft64oracle_f64,
    backend_ref = crate::NTT4x30CIOracle,
    backend_test = crate::FFT64CIOracle,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64oracle_quad,
    backend_ref = crate::NTT4x30CIOracle,
    backend_test = crate::FFT64CIOracle,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30oracle_f64,
    backend_ref = crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30oracle_quad,
    backend_ref = crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}
