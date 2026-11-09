//! The `poulpy-blas` reference compositions, on both rings of every runtime.

#[cfg(feature = "enable-cuda")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Cuda);
#[cfg(feature = "enable-hip")]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Hip);
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
::poulpy_blas::impl_blas_reference_full!(crate::NTT4x30Wgpu);
