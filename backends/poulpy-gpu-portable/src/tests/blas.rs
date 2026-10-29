//! BLAS tests: parity of the GPU types against `NTT4x30Portable` on both rings, bit for bit, also at radix 54 and at
//! quad precision, and the radix helpers of the products on both types. The conjugate-invariant suite takes a host
//! backend, so on a GPU type the parity suites cover that ring.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic functions. This file only instantiates the
//! parity suite via `blas_parity_test_suite!` and the radix helpers via `blas_max_base2k_test_suite!`. There should be
//! no hand-written test here.

use poulpy_blas::{blas_max_base2k_test_suite, blas_parity_test_suite};

type Gpu = crate::NTT4x30Gpu<super::Runtime, poulpy_hal::layouts::Standard>;
type GpuCI = crate::NTT4x30Gpu<super::Runtime, poulpy_hal::layouts::ConjugateInvariant>;

blas_parity_test_suite! {
    mod blas_parity_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_base2k54,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 54 on the conjugate-invariant ring at N = 32, whose products are as long as the standard ring's at N = 64.
blas_parity_test_suite! {
    mod blas_parity_ci_base2k54,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

// The radix helpers on both rings. 32 slots take `n = 32` on the conjugate-invariant ring.
blas_max_base2k_test_suite! {
    mod blas_max_base2k,
    backend = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_max_base2k_test_suite! {
    mod blas_max_base2k_ci,
    backend = super::GpuCI,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 24, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}
