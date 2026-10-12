//! BLAS tests for the portable backends: parity of each type against
//! its same-family oracle, bit for bit, the NTT types' also at radix 57 and
//! the standard FFT64 type's also at `N = 32`, and the conjugate-invariant
//! suite on the conjugate-invariant types.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`,
//! and the conjugate-invariant suite via `blas_conjugate_invariant_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::{blas_conjugate_invariant_test_suite, blas_parity_test_suite};

blas_parity_test_suite! {
    mod blas_parity_fft64portable_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64portable_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_fft64portable_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30portable_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30portable_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 57, where `matmul` sums past the terms the kernel sums between normalizations, and copies of one block overflow
// a kernel that normalizes only at the end: an NTT-family type registers a parameter set at base2k >= 57.
blas_parity_test_suite! {
    mod blas_parity_ntt4x30portable_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

// N = 32: at log_d 2, one matrix fills the 16 slots.
blas_parity_test_suite! {
    mod blas_parity_fft64portable_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portable_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portable_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portable_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portable_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portable_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 57 on the conjugate-invariant ring at N = 32, whose products are as long as the standard ring's at N = 64.
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portable_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64portable,
    backend = crate::FFT64CIPortable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30portable,
    backend = crate::NTT4x30CIPortable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);
