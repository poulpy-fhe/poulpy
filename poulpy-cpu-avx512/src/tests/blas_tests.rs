//! BLAS tests for the AVX-512 backends: parity of each type, Rayon
//! included, against its same-family oracle (NTT4x30Oracle for IFMA), bit for
//! bit, the NTT types' also at radix 57 and the standard FFT64 types' also at
//! `N = 32`, and the conjugate-invariant suite on the conjugate-invariant types.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`,
//! and the conjugate-invariant suite via `blas_conjugate_invariant_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::{blas_conjugate_invariant_test_suite, blas_parity_test_suite};

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

// Radix 57, where `matmul` sums more terms than the kernel sums between normalizations: an NTT-family type registers a
// parameter set at base2k >= 57.
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

// N = 32: at log_d 2, one matrix fills the 16 slots.
blas_parity_test_suite! {
    mod blas_parity_fft64avx512_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
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

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ntt4x30avx512rayon_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Avx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_fft64avx512rayon_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Avx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
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

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifma_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42Ifma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ntt3x42ifmarayon_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT3x42IfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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

// Radix 57 on the conjugate-invariant ring at N = 32, whose products are as long as the standard ring's at N = 64.
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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

#[cfg(feature = "enable-rayon")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30avx512rayon_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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

#[cfg(feature = "enable-ifma")]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifma_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_parity_test_suite! {
    mod blas_parity_ci_ntt3x42ifmarayon_base2k57,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 57, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64avx512,
    backend = crate::FFT64CIAvx512,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30avx512,
    backend = crate::NTT4x30CIAvx512,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

#[cfg(feature = "enable-rayon")]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64avx512rayon,
    backend = crate::FFT64CIAvx512Rayon,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

#[cfg(feature = "enable-rayon")]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30avx512rayon,
    backend = crate::NTT4x30CIAvx512Rayon,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

#[cfg(feature = "enable-ifma")]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt3x42ifma,
    backend = crate::NTT3x42CIIfma,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt3x42ifmarayon,
    backend = crate::NTT3x42CIIfmaRayon,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);
