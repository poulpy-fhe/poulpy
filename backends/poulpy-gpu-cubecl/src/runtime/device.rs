//! One CubeCL client per backend type.
//!
//! `Module::new` and `Backend::alloc_bytes` take no device argument, so a
//! backend type owns exactly one device for the life of the process. Each
//! CubeCL runtime gets its own marker, which lets several of them run side by
//! side on the same card.

use cubecl::prelude::Client;
#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
use cubecl::{Device, RuntimeId};

/// A CubeCL runtime a backend type is bound to.
pub trait GpuRuntime: Send + Sync + 'static {
    /// Name used in logs and benchmark identifiers.
    const NAME: &'static str;

    /// The client of this runtime's device, created on first use.
    fn client() -> &'static Client;
}

#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
fn first_device(name: &str, accept: impl Fn(RuntimeId) -> bool) -> Client {
    let index: usize = std::env::var("POULPY_GPU_DEVICE")
        .ok()
        .map(|v| v.parse().expect("POULPY_GPU_DEVICE must be a device index"))
        .unwrap_or(0);
    let devices: Vec<Device> = Device::enumerate_all()
        .into_iter()
        .filter(|device| accept(device.runtime()))
        .collect();
    let device = devices
        .get(index)
        .unwrap_or_else(|| panic!("no {name} device at index {index} ({} found)", devices.len()));
    device.client()
}

#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
macro_rules! gpu_runtime {
    ($(#[$attr:meta])* $marker:ident, $name:literal, $accept:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $marker;

        impl GpuRuntime for $marker {
            const NAME: &'static str = $name;

            fn client() -> &'static Client {
                static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
                CLIENT.get_or_init(|| first_device($name, $accept))
            }
        }
    };
}

#[cfg(feature = "enable-cuda")]
gpu_runtime!(
    /// CubeCL's CUDA runtime.
    Cuda,
    "cuda",
    |id| matches!(id, RuntimeId::Cuda)
);

#[cfg(feature = "enable-hip")]
gpu_runtime!(
    /// CubeCL's HIP runtime.
    Hip,
    "hip",
    |id| matches!(id, RuntimeId::Hip)
);

#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
gpu_runtime!(
    /// CubeCL's wgpu runtime: Metal, Vulkan and the other wgpu targets.
    Wgpu,
    "wgpu",
    |id| matches!(id, RuntimeId::Wgpu)
);
