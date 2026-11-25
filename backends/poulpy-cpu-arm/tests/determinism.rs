//! CKKS determinism fixtures of the NEON backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(all(feature = "enable-neon", feature = "enable-ckks"))]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64neon, backend = poulpy_cpu_arm::FFT64Neon, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64neon, backend = poulpy_cpu_arm::FFT64Neon);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30neon, backend = poulpy_cpu_arm::NTT4x30Neon, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30neon, backend = poulpy_cpu_arm::NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_fft64neonrayon, backend = poulpy_cpu_arm::FFT64NeonRayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64neonrayon, backend = poulpy_cpu_arm::FFT64NeonRayon);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30neonrayon, backend = poulpy_cpu_arm::NTT4x30NeonRayon, f32);

#[cfg(feature = "enable-rayon")]
conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30neonrayon, backend = poulpy_cpu_arm::NTT4x30NeonRayon);
