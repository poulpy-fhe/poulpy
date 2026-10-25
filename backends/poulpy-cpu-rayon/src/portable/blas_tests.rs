//! BLAS tests for the portable Rayon backends: parity of each type against
//! its same-family oracle, bit for bit, the NTT types' also at radix 54 and
//! the standard FFT64 type's also at `N = 32`, the conjugate-invariant
//! suite on the conjugate-invariant types, and the radix helpers of the products
//! on every type.
//!
//! All test logic lives in `poulpy_blas::test_suite` as backend-generic
//! functions; this file only instantiates the parity suite for each concrete
//! `(reference, backend, scalar, params)` combination via `blas_parity_test_suite!`,
//! the conjugate-invariant suite via `blas_conjugate_invariant_test_suite!` and the
//! radix helpers via `blas_max_base2k_test_suite!`.
//! There should be no hand-written test here.

use poulpy_blas::{blas_conjugate_invariant_test_suite, blas_max_base2k_test_suite, blas_parity_test_suite};

blas_parity_test_suite! {
    mod blas_parity_fft64portablerayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_fft64portablerayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64PortableRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_fft64portablerayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30portablerayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ntt4x30portablerayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30PortableRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 54, the largest in practice.
blas_parity_test_suite! {
    mod blas_parity_ntt4x30portablerayon_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

// N = 32: at log_d 2, one matrix fills the 16 slots.
blas_parity_test_suite! {
    mod blas_parity_fft64portablerayon_n32,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portablerayon_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portablerayon_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortableRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

blas_parity_test_suite! {
    mod blas_parity_ci_fft64portablerayon_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portablerayon_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portablerayon_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortableRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
}

// Radix 54 on the conjugate-invariant ring at N = 32, whose products are as long as the standard ring's at N = 64.
blas_parity_test_suite! {
    mod blas_parity_ci_ntt4x30portablerayon_base2k54,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 16, base2k: 54, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_conjugate_invariant_test_suite!(
    mod blas_ci_fft64portablerayon,
    backend = crate::FFT64CIPortableRayon,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
);

blas_conjugate_invariant_test_suite!(
    mod blas_ci_ntt4x30portablerayon,
    backend = crate::NTT4x30CIPortableRayon,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
);

// The radix helpers on a type of each family, on both rings; 32 slots take `n = 32` on the conjugate-invariant ring.
blas_max_base2k_test_suite! {
    mod blas_max_base2k_fft64portablerayon,
    backend = crate::FFT64PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_max_base2k_test_suite! {
    mod blas_max_base2k_ntt4x30portablerayon,
    backend = crate::NTT4x30PortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}

blas_max_base2k_test_suite! {
    mod blas_max_base2k_ci_fft64portablerayon,
    backend = crate::FFT64CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 24, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

blas_max_base2k_test_suite! {
    mod blas_max_base2k_ci_ntt4x30portablerayon,
    backend = crate::NTT4x30CIPortableRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 32, hw: 24, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
}
