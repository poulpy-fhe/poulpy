use crate::FFT64RefBackend;
use crate::NTT4x30RefBackend;
use crate::ring::CpuRing;

use crate::{FFT64CIRef, FFT64Ref, NTT4x30CIRef, NTT4x30Ref};
use poulpy_ckks::{
    impl_ckks_add_reference, impl_ckks_conjugate_reference, impl_ckks_copy_reference, impl_ckks_dft_reference,
    impl_ckks_encapsulated_mod_up_reference, impl_ckks_encryption_reference, impl_ckks_imag_reference, impl_ckks_mul_reference,
    impl_ckks_neg_reference, impl_ckks_plaintext_reference, impl_ckks_pow2_reference, impl_ckks_rotate_reference,
    impl_ckks_sub_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(FFT64CIRef);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30CIRef);
impl_ckks_conjugate_reference!(FFT64Ref);
impl_ckks_conjugate_reference!(FFT64CIRef);
impl_ckks_conjugate_reference!(NTT4x30Ref);
impl_ckks_conjugate_reference!(NTT4x30CIRef);
impl_ckks_copy_reference!(FFT64Ref);
impl_ckks_copy_reference!(FFT64CIRef);
impl_ckks_copy_reference!(NTT4x30Ref);
impl_ckks_copy_reference!(NTT4x30CIRef);
impl_ckks_encryption_reference!(FFT64Ref);
impl_ckks_encryption_reference!(FFT64CIRef);
impl_ckks_encryption_reference!(NTT4x30Ref);
impl_ckks_encryption_reference!(NTT4x30CIRef);
impl_ckks_imag_reference!(FFT64Ref);
impl_ckks_imag_reference!(FFT64CIRef);
impl_ckks_imag_reference!(NTT4x30Ref);
impl_ckks_imag_reference!(NTT4x30CIRef);
impl_ckks_mul_reference!(FFT64Ref);
impl_ckks_mul_reference!(FFT64CIRef);
impl_ckks_mul_reference!(NTT4x30Ref);
impl_ckks_mul_reference!(NTT4x30CIRef);
impl_ckks_neg_reference!(FFT64Ref);
impl_ckks_neg_reference!(FFT64CIRef);
impl_ckks_neg_reference!(NTT4x30Ref);
impl_ckks_neg_reference!(NTT4x30CIRef);
impl_ckks_pow2_reference!(FFT64Ref);
impl_ckks_pow2_reference!(FFT64CIRef);
impl_ckks_pow2_reference!(NTT4x30Ref);
impl_ckks_pow2_reference!(NTT4x30CIRef);
impl_ckks_rotate_reference!(FFT64Ref);
impl_ckks_rotate_reference!(FFT64CIRef);
impl_ckks_rotate_reference!(NTT4x30Ref);
impl_ckks_rotate_reference!(NTT4x30CIRef);
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

crate::impl_ckks_encoding!(FFT64Ref);
crate::impl_ckks_encoding!(FFT64CIRef);
crate::impl_ckks_paco_coeff_encoding!(FFT64Ref);
crate::impl_ckks_paco_coeff_encoding!(FFT64CIRef);
crate::impl_ckks_ship_coeff_encoding!(FFT64Ref);
crate::impl_ckks_ship_coeff_encoding!(FFT64CIRef);
crate::impl_ckks_encoding!(NTT4x30Ref);
crate::impl_ckks_encoding!(NTT4x30CIRef);
crate::impl_ckks_paco_coeff_encoding!(NTT4x30Ref);
crate::impl_ckks_paco_coeff_encoding!(NTT4x30CIRef);
crate::impl_ckks_ship_coeff_encoding!(NTT4x30Ref);
crate::impl_ckks_ship_coeff_encoding!(NTT4x30CIRef);
impl_ckks_add_reference!(FFT64Ref);
impl_ckks_add_reference!(FFT64CIRef);
impl_ckks_add_reference!(NTT4x30Ref);
impl_ckks_add_reference!(NTT4x30CIRef);
impl_ckks_sub_reference!(FFT64Ref);
impl_ckks_sub_reference!(FFT64CIRef);
impl_ckks_sub_reference!(NTT4x30Ref);
impl_ckks_sub_reference!(NTT4x30CIRef);
impl_ckks_plaintext_reference!(FFT64Ref);
impl_ckks_plaintext_reference!(FFT64CIRef);
impl_ckks_plaintext_reference!(NTT4x30Ref);
impl_ckks_plaintext_reference!(NTT4x30CIRef);
impl_ckks_dft_reference!(FFT64Ref);
impl_ckks_dft_reference!(FFT64CIRef);
impl_ckks_dft_reference!(NTT4x30Ref);
impl_ckks_dft_reference!(NTT4x30CIRef);
