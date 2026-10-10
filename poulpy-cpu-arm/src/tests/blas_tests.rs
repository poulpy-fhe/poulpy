//! BLAS parity tests for the NEON backends: each type, Rayon included, against
//! its same-family oracle, bit for bit.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::blas_parity_test_suite;

blas_parity_test_suite! {
    mod blas_parity_fft64neon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64neon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Neon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_fft64neon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30neon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30neon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Neon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64neonrayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64NeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64neonrayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64NeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64neonrayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64NeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30neonrayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30NeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30neonrayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30NeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64neon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64neon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64neon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30neon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30neon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CINeon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64neonrayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64neonrayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64neonrayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30neonrayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30neonrayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CINeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}
