//! CKKS determinism fixtures of the portable Rayon backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(feature = "enable-ckks")]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64portablerayon, backend = poulpy_cpu_rayon::FFT64PortableRayon, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64portablerayon, backend = poulpy_cpu_rayon::FFT64PortableRayon);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30portablerayon, backend = poulpy_cpu_rayon::NTT4x30PortableRayon, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30portablerayon, backend = poulpy_cpu_rayon::NTT4x30PortableRayon);
