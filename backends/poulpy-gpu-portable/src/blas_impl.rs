//! The `poulpy-blas` reference compositions, on both rings of every runtime.

#[cfg(feature = "enable-cuda")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Cuda);
#[cfg(feature = "enable-cuda")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Cuda<::poulpy_hal::layouts::ConjugateInvariant>);
#[cfg(feature = "enable-hip")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Hip);
#[cfg(feature = "enable-hip")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Hip<::poulpy_hal::layouts::ConjugateInvariant>);
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Wgpu);
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Wgpu<::poulpy_hal::layouts::ConjugateInvariant>);
