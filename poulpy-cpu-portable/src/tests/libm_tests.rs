//! libm tests: the `poulpy_libm::test_suite` precision and budget checks on the
//! NTT backends of this crate.
//!
//! All test logic lives in `poulpy_libm::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `ckks_libm_backend_test_suite!`.
//! There should be no hand-written test here.

use poulpy_libm::ckks_libm_backend_test_suite;

ckks_libm_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Portable,
    scalar = f64,
    encoder = crate::ckks_encoding::EncodingFFTTable<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);
