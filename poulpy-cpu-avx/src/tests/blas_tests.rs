//! BLAS tests for the AVX backends: parity of each type, Rayon
//! included, against its same-family oracle, bit for bit, the NTT types'
//! also at radix 54 and the standard FFT64 types' also at `N = 32`, and the
//! conjugate-invariant suite on the conjugate-invariant types.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`,
//! and the conjugate-invariant suite via `blas_conjugate_invariant_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::{blas_conjugate_invariant_test_suite, blas_parity_test_suite};

blas_parity_test_suite! {
    mod blas_parity_fft64avx_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64avx_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_fft64avx_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 54, the largest in practice.
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

// N = 32: at log_d 2, one matrix fills the 16 slots.
blas_parity_test_suite! {
    mod blas_parity_fft64avx_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avxrayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avxrayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64AvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avxrayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avxrayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avxrayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avxrayon_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avxrayon_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 54 on the conjugate-invariant ring at N = 32, whose products are as long as the standard ring's at N = 64.
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avxrayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avxrayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avxrayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avxrayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avxrayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avxrayon_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64avx,
    backend = crate::FFT64CIAvx,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30avx,
    backend = crate::NTT4x30CIAvx,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

#[cfg(feature = "enable-rayon")]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64avxrayon,
    backend = crate::FFT64CIAvxRayon,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

#[cfg(feature = "enable-rayon")]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30avxrayon,
    backend = crate::NTT4x30CIAvxRayon,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);
