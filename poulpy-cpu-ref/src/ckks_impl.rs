use crate::{FFT64Ref, NTT4x30Ref};
use poulpy_ckks::{
    impl_ckks_add_defaults, impl_ckks_bootstrap_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults,
    impl_ckks_dft_defaults, impl_ckks_encapsulated_mod_up_default, impl_ckks_encryption_defaults, impl_ckks_eval_mod_defaults,
    impl_ckks_imag_defaults, impl_ckks_mul_defaults, impl_ckks_neg_defaults, impl_ckks_plaintext_defaults,
    impl_ckks_pow2_defaults, impl_ckks_rotate_defaults, impl_ckks_sub_defaults,
};

impl_ckks_encapsulated_mod_up_default!(FFT64Ref);
impl_ckks_encapsulated_mod_up_default!(NTT4x30Ref);
impl_ckks_conjugate_defaults!(FFT64Ref);
impl_ckks_conjugate_defaults!(NTT4x30Ref);
impl_ckks_copy_defaults!(FFT64Ref);
impl_ckks_copy_defaults!(NTT4x30Ref);
impl_ckks_encryption_defaults!(FFT64Ref);
impl_ckks_encryption_defaults!(NTT4x30Ref);
impl_ckks_imag_defaults!(FFT64Ref);
impl_ckks_imag_defaults!(NTT4x30Ref);
impl_ckks_mul_defaults!(FFT64Ref);
impl_ckks_mul_defaults!(NTT4x30Ref);
impl_ckks_neg_defaults!(FFT64Ref);
impl_ckks_neg_defaults!(NTT4x30Ref);
impl_ckks_pow2_defaults!(FFT64Ref);
impl_ckks_pow2_defaults!(NTT4x30Ref);
impl_ckks_rotate_defaults!(FFT64Ref);
impl_ckks_rotate_defaults!(NTT4x30Ref);
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
impl_ckks_add_defaults!(FFT64Ref);
impl_ckks_add_defaults!(NTT4x30Ref);
impl_ckks_sub_defaults!(FFT64Ref);
impl_ckks_sub_defaults!(NTT4x30Ref);
impl_ckks_plaintext_defaults!(FFT64Ref);
impl_ckks_plaintext_defaults!(NTT4x30Ref);
impl_ckks_dft_defaults!(FFT64Ref);
impl_ckks_dft_defaults!(NTT4x30Ref);
impl_ckks_eval_mod_defaults!(FFT64Ref);
impl_ckks_eval_mod_defaults!(NTT4x30Ref);
impl_ckks_bootstrap_defaults!(FFT64Ref);
impl_ckks_bootstrap_defaults!(NTT4x30Ref);
