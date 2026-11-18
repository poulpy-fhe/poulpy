//! CKKS determinism fixtures of the portable backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(feature = "enable-ckks")]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64portable, backend = poulpy_cpu_portable::FFT64Portable, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64portable, backend = poulpy_cpu_portable::FFT64Portable);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30portable, backend = poulpy_cpu_portable::NTT4x30Portable, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30portable, backend = poulpy_cpu_portable::NTT4x30Portable);
