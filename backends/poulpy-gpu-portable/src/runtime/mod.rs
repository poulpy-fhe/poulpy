//! Device runtime: the CubeCL client of each backend type, device buffers and
//! their views, and the helpers every kernel launch goes through.

pub(crate) mod access;
mod buffer;
mod device;
mod launch;
mod replay;

pub(crate) use buffer::{GpuBuf, GpuMut, GpuRef, read_at, write_at};
#[cfg(feature = "enable-cuda")]
pub use device::Cuda;
pub use device::GpuRuntime;
#[cfg(feature = "enable-hip")]
pub use device::Hip;
#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
pub use device::Wgpu;
pub(crate) use device::on_stream;
pub(crate) use device::{NttGeometry, RuntimeDevice};
pub use launch::HostEvents;
#[cfg(test)]
pub(crate) use launch::profile;
pub(crate) use launch::{
    Binding, Bound, BufferArgs, HostEvent, Label, Trace, Tracer, cubes, host_event, launch, launched, replayed, trace_launch,
    tracing,
};
#[cfg(test)]
#[allow(unused_imports, reason = "read by the tests of an enabled runtime")]
pub(crate) use launch::{launch_breakdown, launch_counters, reset_launch_counters};
pub(crate) use replay::{GraphRecording, record};
