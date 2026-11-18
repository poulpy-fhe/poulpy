//! CKKS determinism fixtures of the oracle backend types, on the standard and
//! conjugate-invariant rings.

#![cfg(feature = "enable-ckks")]

use poulpy_ckks::{ckks_determinism_test_suite, conjugate_invariant_determinism_test_suite};

ckks_determinism_test_suite!(mod ckks_determinism_fft64oracle, backend = poulpy_cpu_oracle::FFT64Oracle, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_fft64oracle, backend = poulpy_cpu_oracle::FFT64Oracle);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30oracle, backend = poulpy_cpu_oracle::NTT4x30Oracle, f32);

conjugate_invariant_determinism_test_suite!(mod ckks_ci_determinism_ntt4x30oracle, backend = poulpy_cpu_oracle::NTT4x30Oracle);
