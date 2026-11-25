//! CKKS determinism fixtures of the AVX-512 backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(all(feature = "enable-avx512f", feature = "enable-ckks"))]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64avx512, backend = poulpy_cpu_avx512::FFT64Avx512, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64avx512, backend = poulpy_cpu_avx512::FFT64Avx512);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avx512, backend = poulpy_cpu_avx512::NTT4x30Avx512, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30avx512, backend = poulpy_cpu_avx512::NTT4x30Avx512);

#[cfg(feature = "enable-ifma")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt3x42ifma, backend = poulpy_cpu_avx512::NTT3x42Ifma, f32);

#[cfg(feature = "enable-ifma")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt3x42ifma, backend = poulpy_cpu_avx512::NTT3x42Ifma);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_fft64avx512rayon, backend = poulpy_cpu_avx512::FFT64Avx512Rayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64avx512rayon, backend = poulpy_cpu_avx512::FFT64Avx512Rayon);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avx512rayon, backend = poulpy_cpu_avx512::NTT4x30Avx512Rayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30avx512rayon, backend = poulpy_cpu_avx512::NTT4x30Avx512Rayon);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
ckks_determinism_test_suite!(mod ckks_determinism_ntt3x42ifmarayon, backend = poulpy_cpu_avx512::NTT3x42IfmaRayon, f32);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt3x42ifmarayon, backend = poulpy_cpu_avx512::NTT3x42IfmaRayon);
