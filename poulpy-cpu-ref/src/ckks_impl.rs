use crate::FFT64RefBackend;
use crate::NTT4x30RefBackend;
use crate::ring::CpuRing;

use crate::{FFT64CIRef, FFT64Ref, NTT4x30CIRef, NTT4x30Ref};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(FFT64CIRef);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30CIRef);
// The reference backends have no accelerated transform, so they select the
// generic scalar table for every precision at once.
impl<R: CpuRing, F> crate::ckks_encoding::CKKSEncodingTransform<F> for FFT64RefBackend<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

impl<R: CpuRing, F> crate::ckks_encoding::CKKSEncodingTransform<F> for NTT4x30RefBackend<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

crate::impl_cpu_ckks_defaults!(crate::FFT64Ref);
crate::impl_cpu_ckks_defaults!(crate::FFT64CIRef);
crate::impl_cpu_ckks_defaults!(crate::NTT4x30Ref);
crate::impl_cpu_ckks_defaults!(crate::NTT4x30CIRef);

#[cfg(any(test, feature = "enable-test-suite"))]
mod comparison {
    use crate::test_suite::ControlledSamplingFFT64Ref;
    poulpy_ckks::impl_ckks_plaintext_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_copy_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_add_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_sub_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_neg_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_pow2_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_imag_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_rotate_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_conjugate_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_encryption_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_mul_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_polynomial_evaluation_reference!(ControlledSamplingFFT64Ref);
    poulpy_ckks::impl_ckks_eval_mod_reference!(ControlledSamplingFFT64Ref);
}
