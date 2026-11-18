//! CKKS determinism fixtures of the AVX backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(all(feature = "enable-avx", feature = "enable-ckks"))]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64avx, backend = poulpy_cpu_avx::FFT64Avx, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64avx, backend = poulpy_cpu_avx::FFT64Avx);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avx, backend = poulpy_cpu_avx::NTT4x30Avx, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30avx, backend = poulpy_cpu_avx::NTT4x30Avx);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_fft64avxrayon, backend = poulpy_cpu_avx::FFT64AvxRayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64avxrayon, backend = poulpy_cpu_avx::FFT64AvxRayon);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avxrayon, backend = poulpy_cpu_avx::NTT4x30AvxRayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30avxrayon, backend = poulpy_cpu_avx::NTT4x30AvxRayon);
