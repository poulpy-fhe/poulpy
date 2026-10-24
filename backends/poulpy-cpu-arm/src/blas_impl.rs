//! The `poulpy-blas` reference compositions.

::poulpy_blas::impl_blas_reference_full!(crate::FFT64Neon);
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Neon);
::poulpy_blas::impl_blas_reference_full!(crate::FFT64CINeon);
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30CINeon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30CINeonRayon);
