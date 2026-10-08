//! What every kernel launch shares: the binding of a view, dispatch sizing and
//! the launch counters.

use std::sync::atomic::{AtomicU64, Ordering};

use cubecl::{prelude::*, server::Handle};

/// A view as a kernel sees it: the whole allocation, and where the view starts.
#[derive(Clone)]
pub(crate) struct Binding {
    pub(crate) handle: Handle,
    /// Bytes reserved by the allocation.
    pub(crate) reserved: usize,
    /// Byte offset of the view in the allocation.
    pub(crate) offset: usize,
}

impl Binding {
    /// The allocation as a slice of `T`.
    pub(crate) fn arg<T>(&self) -> BufferArg {
        // SAFETY: the handle spans `reserved` bytes, a whole number of `T`.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.reserved / size_of::<T>()) }
    }

    /// Index of the view's first element in [`Self::arg`].
    pub(crate) fn base<T>(&self) -> u32 {
        debug_assert!(self.offset.is_multiple_of(size_of::<T>()));
        u32::try_from(self.offset / size_of::<T>()).expect("device view starts beyond the 32-bit element range")
    }
}

/// Dispatch covering `units` threads in groups of `dim`.
pub(crate) fn cubes(client: &Client, units: usize, dim: u32) -> CubeCount {
    let groups = u32::try_from(units.div_ceil(dim as usize)).expect("dispatch exceeds the 32-bit group range");
    cubecl::server::CubeCountSelection::new(client, groups.max(1)).cube_count()
}

static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

/// Kernel launches and the bytes they were sized to read plus write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaunchCounters {
    pub launches: u64,
    pub bytes: u64,
}

/// Records one launch moving `bytes` of device memory.
pub(crate) fn launched(bytes: usize) {
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Launches issued by every backend type of this crate since the last reset.
pub fn launch_counters() -> LaunchCounters {
    LaunchCounters {
        launches: LAUNCHES.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
    }
}

/// Resets [`launch_counters`].
pub fn reset_launch_counters() {
    LAUNCHES.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}
