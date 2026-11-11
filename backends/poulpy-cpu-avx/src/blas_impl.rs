//! The `poulpy-blas` reference compositions.

::poulpy_blas::impl_blas_reference_full!(crate::FFT64Avx);
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Avx);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30AvxRayon);
