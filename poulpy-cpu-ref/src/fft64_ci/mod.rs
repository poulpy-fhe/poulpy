//! Portable conjugate-invariant FFT64 backend.
//!
//! [`FFT64CIRef`] serves the conjugate invariant ring with its own module
//! handle and plans. Every ring-independent kernel is forwarded to
//! [`FFT64Ref`](crate::FFT64Ref); the conjugate invariant pre- and
//! post-processing lives in the FFT plans the handle builds.

mod hal;
mod module;

/// Portable conjugate-invariant FFT64 backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CIRef;

crate::forward_znx_kernels!(FFT64CIRef => crate::FFT64Ref);
crate::forward_fft64_kernels!(FFT64CIRef => crate::FFT64Ref);

#[cfg(feature = "enable-core")]
mod core_impl {
    use super::FFT64CIRef;

    poulpy_core::impl_glwe_tensoring_reference!(FFT64CIRef);
    poulpy_core::impl_gglwe_product_digits_strided_reference!(FFT64CIRef);
    crate::impl_cpu_core_defaults!(FFT64CIRef, fft64);
}
