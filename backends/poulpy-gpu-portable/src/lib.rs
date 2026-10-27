//! GPU backends for `poulpy-hal`, on CubeCL.
//!
//! One arithmetic subfamily, NTT4x30, with the prime set of the CPU backends:
//! results are bit-identical to `NTT4x30Portable`. The implementation is
//! written once on [`NTT4x30Gpu`]; each CubeCL runtime gets its own backend
//! type, so several runtimes can run side by side in one process.
//!
//! # Features
//!
//! - `enable-cuda`: [`NTT4x30Cuda`], on CubeCL's CUDA runtime.
//! - `enable-hip`: [`NTT4x30Hip`], on CubeCL's HIP runtime.
//! - `enable-metal`, `enable-vulkan`, `enable-wgpu`: [`NTT4x30Wgpu`], on CubeCL's wgpu runtime with
//!   the Metal, Vulkan or WebGPU shader compiler.
//! - `enable-core`, `enable-ckks`: the `poulpy-core` and `poulpy-ckks` families on every backend type.
//!
//! # Tuning
//!
//! A module's [`GpuHandle`], reached with [`handle`], carries the tuning of its
//! kernels the device decides: [`GpuHandle::set_convolution_capacity_step`].

mod backend;
#[cfg(feature = "enable-ckks")]
mod ckks_add_sub;
#[cfg(feature = "enable-ckks")]
mod ckks_impl;
#[cfg(feature = "enable-ckks")]
mod ckks_round;
#[cfg(feature = "enable-core")]
mod core_impl;
mod hal_impl;
mod module;
mod ntt4x30;
mod replay;
mod runtime;
mod znx;

#[cfg(test)]
mod tests;

pub use backend::NTT4x30Gpu;
pub use module::{GpuHandle, MAX_CAPACITY_STEP, handle};
pub use ntt4x30::AutomorphismPlan;
pub use replay::Replay;
#[cfg(feature = "enable-cuda")]
pub use runtime::Cuda;
#[cfg(feature = "enable-hip")]
pub use runtime::Hip;
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
pub use runtime::Wgpu;
pub use runtime::{GpuRuntime, HostEvents};
#[cfg(test)]
#[allow(unused_imports, reason = "read by the tests of an enabled runtime")]
pub(crate) use runtime::{launch_breakdown, launch_counters, reset_launch_counters};

/// NTT4x30 on CubeCL's CUDA runtime.
#[cfg(feature = "enable-cuda")]
pub type NTT4x30Cuda<R = poulpy_hal::layouts::Standard> = NTT4x30Gpu<runtime::Cuda, R>;

/// NTT4x30 on CubeCL's HIP runtime.
#[cfg(feature = "enable-hip")]
pub type NTT4x30Hip<R = poulpy_hal::layouts::Standard> = NTT4x30Gpu<runtime::Hip, R>;

/// NTT4x30 on CubeCL's wgpu runtime: Metal, Vulkan and the other wgpu targets.
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
pub type NTT4x30Wgpu<R = poulpy_hal::layouts::Standard> = NTT4x30Gpu<runtime::Wgpu, R>;
