//! The `poulpy-blas` reference compositions.

::poulpy_blas::impl_blas_reference_full!(crate::FFT64Avx512);
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Avx512);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-ifma")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT3x42Ifma);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
::poulpy_blas::impl_blas_reference_full!(crate::NTT3x42IfmaRayon);
