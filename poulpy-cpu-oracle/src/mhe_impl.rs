//! Registers the generic multiparty compositions for the standard oracles and
//! the controlled-sampling FFT64 oracle.

use crate::{FFT64Oracle, NTT4x30Oracle, test_suite::ControlledSamplingFFT64Oracle};

poulpy_mhe::impl_mhe_reference_full!(FFT64Oracle);
poulpy_mhe::impl_mhe_reference_full!(NTT4x30Oracle);
poulpy_mhe::impl_mhe_reference_full!(ControlledSamplingFFT64Oracle);

#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_fft64oracle, backend = crate::FFT64Oracle);
#[cfg(test)]
poulpy_mhe::mhe_backend_test_suite!(mod mhe_ntt4x30oracle, backend = crate::NTT4x30Oracle);
// The two families draw identical samples from a seed.
#[cfg(test)]
poulpy_mhe::mhe_parity_test_suite!(mod mhe_parity_oracle, backend_ref = crate::FFT64Oracle, backend_test = crate::NTT4x30Oracle);
