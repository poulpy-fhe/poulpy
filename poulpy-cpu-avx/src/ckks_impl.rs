use crate::{FFT64Avx, NTT120Avx};
use poulpy_ckks::{
    impl_ckks_conjugate_defaults, impl_ckks_copy_defaults, impl_ckks_encryption_defaults, impl_ckks_imag_defaults,
    impl_ckks_mul_defaults, impl_ckks_neg_defaults, impl_ckks_pow2_defaults, impl_ckks_rescale_defaults,
    impl_ckks_rotate_default,
};

impl_ckks_conjugate_defaults!(FFT64Avx);
impl_ckks_conjugate_defaults!(NTT120Avx);
impl_ckks_copy_defaults!(FFT64Avx);
impl_ckks_copy_defaults!(NTT120Avx);
impl_ckks_encryption_defaults!(FFT64Avx);
impl_ckks_encryption_defaults!(NTT120Avx);
impl_ckks_imag_defaults!(FFT64Avx);
impl_ckks_imag_defaults!(NTT120Avx);
impl_ckks_mul_defaults!(FFT64Avx);
impl_ckks_mul_defaults!(NTT120Avx);
impl_ckks_neg_defaults!(FFT64Avx);
impl_ckks_neg_defaults!(NTT120Avx);
impl_ckks_pow2_defaults!(FFT64Avx);
impl_ckks_pow2_defaults!(NTT120Avx);
impl_ckks_rescale_defaults!(FFT64Avx);
impl_ckks_rescale_defaults!(NTT120Avx);
impl_ckks_rotate_default!(FFT64Avx);
impl_ckks_rotate_default!(NTT120Avx);
