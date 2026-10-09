//! One CubeCL client per backend type.
//!
//! `Module::new` and `Backend::alloc_bytes` take no device argument, so a
//! backend type owns exactly one device for the life of the process. Each
//! CubeCL runtime gets its own marker, which lets several of them run side by
//! side on the same card.

use cubecl::prelude::Client;
#[cfg(any(
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
use cubecl::{Device, RuntimeId};
#[cfg(any(
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
use cubecl_environment::stream::StreamId;

/// The stream of the backend clients (`"poulpy"` in ASCII).
#[cfg(any(
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
const STREAM: u64 = 0x706f_756c_7079;

/// How a runtime wants the launches and the workgroups of the transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttGeometry {
    /// Most stages one launch runs: a tile has `2^stages` lanes. Up to four
    /// stages, a thread holds a whole tile in registers; beyond, the threads
    /// of a tile hand lanes over through shared memory at barriers.
    pub max_stages: u32,
    /// log2 of the threads a workgroup is filled up to, with whole tiles.
    pub log_threads: u32,
    /// log2 of the least number of tiles in a workgroup.
    pub log_tiles: u32,
}

impl NttGeometry {
    /// A GPU: up to twelve stages per launch, workgroups of 128 threads and 8 tiles at least.
    pub const GPU: Self = Self {
        max_stages: 12,
        log_threads: 7,
        log_tiles: 3,
    };
    /// CubeCL's CPU runtime, where every thread of a workgroup is a task and
    /// a barrier needs them all running: four stages per launch, so that no
    /// launch has a barrier, in workgroups of 8 tiles.
    pub const CPU: Self = Self {
        max_stages: 4,
        log_threads: 3,
        log_tiles: 0,
    };
}

/// A CubeCL runtime a backend type is bound to.
pub trait GpuRuntime: Send + Sync + 'static {
    /// Name used in logs and benchmark identifiers.
    const NAME: &'static str;

    /// Threads per workgroup of the kernels whose threads are independent.
    /// A GPU runs a workgroup on one of its cores at no cost per thread;
    /// CubeCL's CPU runtime gives every thread of a workgroup its own task.
    const GROUP: u32;

    /// The workgroups of the transform.
    const NTT: NttGeometry;

    /// Selects the device of this runtime by its index among the devices
    /// CubeCL enumerates for it; the first one is used otherwise. `false`
    /// when the device is already in use: the choice is made once, before
    /// the first module or buffer of a backend type on this runtime.
    fn select_device(index: usize) -> bool;

    /// The client of this runtime's device, created on first use.
    #[doc(hidden)]
    fn client() -> &'static Client;
}

/// Runs `task` with the client's stream as the current one of the thread,
/// for the client calls that act on the current stream and not on the
/// client's.
#[cfg(any(
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
pub(crate) fn on_stream<T>(task: impl FnOnce() -> T) -> T {
    StreamId { value: STREAM }.executes(task)
}

#[cfg(not(any(
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
)))]
pub(crate) fn on_stream<T>(task: impl FnOnce() -> T) -> T {
    task()
}

#[cfg(any(
    feature = "enable-cpu",
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
    feature = "enable-cpu",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan",
    feature = "enable-wgpu"
))]
macro_rules! gpu_runtime {
    ($(#[$attr:meta])* $marker:ident, $name:literal, $group:literal, $ntt:expr, $accept:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $marker;

        impl GpuRuntime for $marker {
            const NAME: &'static str = $name;
            const GROUP: u32 = $group;
            const NTT: NttGeometry = $ntt;

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
    256,
    NttGeometry::GPU,
    |id| matches!(id, RuntimeId::Cuda)
);

#[cfg(feature = "enable-hip")]
gpu_runtime!(
    /// CubeCL's HIP runtime.
    Hip,
    "hip",
    256,
    NttGeometry::GPU,
    |id| matches!(id, RuntimeId::Hip)
);

#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
gpu_runtime!(
    /// CubeCL's wgpu runtime: Metal, Vulkan and the other wgpu targets.
    Wgpu,
    "wgpu",
    256,
    NttGeometry::GPU,
    |id| matches!(id, RuntimeId::Wgpu)
);

#[cfg(feature = "enable-cpu")]
gpu_runtime!(
    /// CubeCL's CPU runtime.
    Cpu,
    "cpu",
    8,
    NttGeometry::CPU,
    |id| matches!(id, RuntimeId::Cpu)
);
