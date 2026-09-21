use crate::{FFT64Ref, NTT4x30Ref};
use poulpy_ckks::{
    impl_ckks_add_reference, impl_ckks_conjugate_reference, impl_ckks_copy_reference, impl_ckks_dft_reference,
    impl_ckks_encapsulated_mod_up_reference, impl_ckks_encryption_reference, impl_ckks_eval_mod_reference,
    impl_ckks_imag_reference, impl_ckks_mul_reference, impl_ckks_neg_reference, impl_ckks_plaintext_reference,
    impl_ckks_polynomial_evaluation_reference, impl_ckks_pow2_reference, impl_ckks_rotate_reference, impl_ckks_sub_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
impl_ckks_conjugate_reference!(FFT64Ref);
impl_ckks_conjugate_reference!(NTT4x30Ref);
impl_ckks_copy_reference!(FFT64Ref);
impl_ckks_copy_reference!(NTT4x30Ref);
impl_ckks_encryption_reference!(FFT64Ref);
impl_ckks_encryption_reference!(NTT4x30Ref);
impl_ckks_imag_reference!(FFT64Ref);
impl_ckks_imag_reference!(NTT4x30Ref);
impl_ckks_mul_reference!(FFT64Ref);
impl_ckks_mul_reference!(NTT4x30Ref);
impl_ckks_neg_reference!(FFT64Ref);
impl_ckks_neg_reference!(NTT4x30Ref);
impl_ckks_pow2_reference!(FFT64Ref);
impl_ckks_pow2_reference!(NTT4x30Ref);
impl_ckks_rotate_reference!(FFT64Ref);
impl_ckks_rotate_reference!(NTT4x30Ref);
// The reference backends have no accelerated transform, so they select the
// generic scalar table for every precision at once.
impl<F> crate::ckks_encoding::CKKSEncodingTransform<F> for FFT64Ref
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

impl<F> crate::ckks_encoding::CKKSEncodingTransform<F> for NTT4x30Ref
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

crate::impl_ckks_encoding!(FFT64Ref);
crate::impl_ckks_paco_coeff_encoding!(FFT64Ref);
crate::impl_ckks_ship_coeff_encoding!(FFT64Ref);
crate::impl_ckks_encoding!(NTT4x30Ref);
crate::impl_ckks_paco_coeff_encoding!(NTT4x30Ref);
crate::impl_ckks_ship_coeff_encoding!(NTT4x30Ref);
impl_ckks_add_reference!(FFT64Ref);
impl_ckks_add_reference!(NTT4x30Ref);
impl_ckks_sub_reference!(FFT64Ref);
impl_ckks_sub_reference!(NTT4x30Ref);
impl_ckks_plaintext_reference!(FFT64Ref);
impl_ckks_plaintext_reference!(NTT4x30Ref);
impl_ckks_dft_reference!(FFT64Ref);
impl_ckks_eval_mod_reference!(FFT64Ref);
impl_ckks_polynomial_evaluation_reference!(FFT64Ref);
impl_ckks_dft_reference!(NTT4x30Ref);
impl_ckks_eval_mod_reference!(NTT4x30Ref);
impl_ckks_polynomial_evaluation_reference!(NTT4x30Ref);

#[cfg(any(test, feature = "enable-test-suite"))]
mod comparison {
    use super::*;
    use crate::test_suite::ControlledSamplingFFT64Ref;
    impl_ckks_plaintext_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_copy_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_add_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_sub_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_neg_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_pow2_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_imag_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_rotate_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_conjugate_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_encryption_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_mul_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_polynomial_evaluation_reference!(ControlledSamplingFFT64Ref);
    impl_ckks_eval_mod_reference!(ControlledSamplingFFT64Ref);
}
