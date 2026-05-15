use crate::{FFT64Ref, NTT120Ref};
use poulpy_ckks::{
    impl_ckks_add_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults, impl_ckks_encryption_defaults,
    impl_ckks_imag_defaults, impl_ckks_maintain_ops_defaults, impl_ckks_mul_defaults, impl_ckks_neg_defaults,
    impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults, impl_ckks_rescale_defaults, impl_ckks_rotate_default,
    impl_ckks_sub_defaults,
};

impl_ckks_conjugate_defaults!(FFT64Ref);
impl_ckks_conjugate_defaults!(NTT120Ref);
impl_ckks_copy_defaults!(FFT64Ref);
impl_ckks_copy_defaults!(NTT120Ref);
impl_ckks_encryption_defaults!(FFT64Ref);
impl_ckks_encryption_defaults!(NTT120Ref);
impl_ckks_imag_defaults!(FFT64Ref);
impl_ckks_imag_defaults!(NTT120Ref);
impl_ckks_mul_defaults!(FFT64Ref);
impl_ckks_mul_defaults!(NTT120Ref);
impl_ckks_neg_defaults!(FFT64Ref);
impl_ckks_neg_defaults!(NTT120Ref);
impl_ckks_pow2_defaults!(FFT64Ref);
impl_ckks_pow2_defaults!(NTT120Ref);
impl_ckks_rescale_defaults!(FFT64Ref);
impl_ckks_rescale_defaults!(NTT120Ref);
impl_ckks_rotate_default!(FFT64Ref);
impl_ckks_rotate_default!(NTT120Ref);
impl_ckks_add_defaults!(FFT64Ref);
impl_ckks_add_defaults!(NTT120Ref);
impl_ckks_sub_defaults!(FFT64Ref);
impl_ckks_sub_defaults!(NTT120Ref);
impl_ckks_plaintext_defaults!(FFT64Ref);
impl_ckks_plaintext_defaults!(NTT120Ref);
impl_ckks_maintain_ops_defaults!(FFT64Ref);
impl_ckks_maintain_ops_defaults!(NTT120Ref);
