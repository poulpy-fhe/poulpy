//! BLAS parity tests for the AVX-512 backends: each type, Rayon included, against
//! its same-family oracle (NTT4x30Oracle for IFMA), bit for bit.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::blas_parity_test_suite;

blas_parity_test_suite! {
    mod blas_parity_fft64avx512_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64avx512_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_fft64avx512_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avx512rayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avx512rayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avx512rayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512rayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512rayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifma_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42Ifma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifma_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42Ifma,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifmarayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42IfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifmarayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42IfmaRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512rayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512rayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_fft64avx512rayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512rayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512rayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifma_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifma_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfma,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifmarayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifmarayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}
