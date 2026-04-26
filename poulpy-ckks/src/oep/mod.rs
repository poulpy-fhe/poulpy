mod add;
mod ckks_impl;
mod conjugate;
mod mul;
mod neg;
mod pow2;
mod pt_znx;
mod rotate;
mod sub;

pub use add::impl_ckks_add_default_methods;
pub use ckks_impl::CKKSImpl;
pub use ckks_impl::impl_ckks_default_methods;
pub use conjugate::impl_ckks_conjugate_default_methods;
pub use mul::impl_ckks_mul_default_methods;
pub use neg::impl_ckks_neg_default_methods;
pub use pow2::impl_ckks_pow2_default_methods;
pub use pt_znx::impl_ckks_pt_znx_default_methods;
pub use rotate::impl_ckks_rotate_default_methods;
pub use sub::impl_ckks_sub_default_methods;

unsafe impl CKKSImpl<poulpy_cpu_ref::FFT64Ref> for poulpy_cpu_ref::FFT64Ref {
    crate::impl_ckks_default_methods!(poulpy_cpu_ref::FFT64Ref);
}

unsafe impl CKKSImpl<poulpy_cpu_ref::NTT120Ref> for poulpy_cpu_ref::NTT120Ref {
    crate::impl_ckks_default_methods!(poulpy_cpu_ref::NTT120Ref);
}

#[cfg(feature = "enable-avx")]
unsafe impl CKKSImpl<poulpy_cpu_avx::FFT64Avx> for poulpy_cpu_avx::FFT64Avx {
    crate::impl_ckks_default_methods!(poulpy_cpu_avx::FFT64Avx);
}

#[cfg(feature = "enable-avx")]
unsafe impl CKKSImpl<poulpy_cpu_avx::NTT120Avx> for poulpy_cpu_avx::NTT120Avx {
    crate::impl_ckks_default_methods!(poulpy_cpu_avx::NTT120Avx);
}
