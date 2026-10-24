//! libm tests: the `poulpy_libm::test_suite` precision and budget checks on the
//! NTT backends of this crate.
//!
//! libm overrides nothing, so the suite checks its compositions once per backend
//! family, on the NTT types at the radix-52 parameter set its precision floors are
//! calibrated on. The FFT64 types run CKKS at radix 19 only, the Rayon twins are
//! bit-exact with their serial types, which their parity suites check, and the
//! libm plans are built in `f64`.
//!
//! All test logic lives in `poulpy_libm::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `libm_backend_test_suite!`.
//! There should be no hand-written test here.

use poulpy_libm::libm_backend_test_suite;

libm_backend_test_suite!(
    mod ntt4x30_neon_f64,
    backend = crate::NTT4x30Neon,
    scalar = f64,
    encoder = crate::FFT64NeonEncodingTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);
