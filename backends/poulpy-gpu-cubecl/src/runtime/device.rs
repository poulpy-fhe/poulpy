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
#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
use cubecl_environment::stream::StreamId;

/// The stream of the backend clients (`"poulpy"` in ASCII).
#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
const STREAM: u64 = 0x706f_756c_7079;

/// A CubeCL runtime a backend type is bound to.
pub trait GpuRuntime: Send + Sync + 'static {
    /// Name used in logs and benchmark identifiers.
    const NAME: &'static str;

    /// Selects the device of this runtime by its index among the devices
    /// CubeCL enumerates for it; the first one is used otherwise. `false`
    /// when the device is already in use: the choice is made once, before
    /// the first module or buffer of a backend type on this runtime.
    fn select_device(index: usize) -> bool;

    /// The client of this runtime's device, created on first use.
    #[doc(hidden)]
    fn client() -> &'static Client;
}

#[cfg(any(
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
fn device_client(name: &str, index: usize, accept: impl Fn(RuntimeId) -> bool) -> Client {
    let devices: Vec<Device> = Device::enumerate_all()
        .into_iter()
        .filter(|device| accept(device.runtime()))
        .collect();
    let device = devices
        .get(index)
        .unwrap_or_else(|| panic!("no {name} device at index {index} ({} found)", devices.len()));
    let mut client = device.client();
    // Every thread reaches the device through this one client. Pinned to its
    // own stream, the launches, the host transfers and the replays of
    // recorded graphs are ordered against each other whichever thread issues them.
    // SAFETY: the client is not shared yet, and the stream is used by nothing else.
    unsafe { client.set_stream(StreamId { value: STREAM }) };
    client
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

            fn select_device(index: usize) -> bool {
                let mut selected = false;
                Self::state().0.get_or_init(|| {
                    selected = true;
                    index
                });
                selected && Self::state().1.get().is_none()
            }

            fn client() -> &'static Client {
                let (index, client) = Self::state();
                client.get_or_init(|| device_client($name, *index.get_or_init(|| 0), $accept))
            }
        }

        impl $marker {
            /// The selected device index and its client, each set once.
            fn state() -> &'static (std::sync::OnceLock<usize>, std::sync::OnceLock<Client>) {
                static STATE: (std::sync::OnceLock<usize>, std::sync::OnceLock<Client>) =
                    (std::sync::OnceLock::new(), std::sync::OnceLock::new());
                &STATE
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
