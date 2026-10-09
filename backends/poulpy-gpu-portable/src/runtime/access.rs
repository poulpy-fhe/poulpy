//! Ordering of device traffic against graph capture and replay.
//!
//! Every thread reaches the device through the one client of its runtime.
//! While a capture window is open on that client, CubeCL records every
//! launch instead of executing it, whichever thread issues it, and refuses
//! reads, host writes and synchronizations. A recording therefore excludes
//! all other device traffic, and so does a replay, which rewrites the
//! temporaries of the recorded body.
//!
//! Launches, transfers, allocations and synchronizations take the lock
//! [shared](shared); a recording and a replay take it [exclusively](exclusive).
//! The lock is process-wide: a capture holds its stream, and on CUDA a
//! concurrent allocation or synchronization on the device aborts it.

use std::{
    cell::Cell,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

static DEVICE: RwLock<()> = RwLock::new(());

thread_local! {
    /// Shared accesses open on this thread.
    static SHARED: Cell<u32> = const { Cell::new(0) };
    /// Whether this thread holds the exclusive access.
    static EXCLUSIVE: Cell<bool> = const { Cell::new(false) };
}

/// A shared access to the device, released on drop.
pub(crate) struct Shared {
    /// Held by the outermost access of the thread.
    _guard: Option<RwLockReadGuard<'static, ()>>,
}

/// The exclusive access to the device, released on drop.
pub(crate) struct Exclusive {
    _guard: RwLockWriteGuard<'static, ()>,
}

/// Takes a shared access. It nests: a thread that already holds an access,
/// shared or exclusive, is not made to wait for a recording that is itself
/// waiting for that thread.
pub(crate) fn shared() -> Shared {
    if EXCLUSIVE.get() {
        return Shared { _guard: None };
    }
    let depth = SHARED.get();
    SHARED.set(depth + 1);
    Shared {
        _guard: (depth == 0).then(|| DEVICE.read().unwrap_or_else(|poisoned| poisoned.into_inner())),
    }
}

/// Takes the exclusive access, waiting for every shared one to be released.
///
/// # Panics
///
/// If this thread already holds an access: a recording or a replay cannot
/// start inside a recorded body or in the middle of a backend call.
pub(crate) fn exclusive() -> Exclusive {
    assert!(
        !EXCLUSIVE.get() && SHARED.get() == 0,
        "a recording or a replay cannot start inside another one"
    );
    let guard = DEVICE.write().unwrap_or_else(|poisoned| poisoned.into_inner());
    EXCLUSIVE.set(true);
    Exclusive { _guard: guard }
}

impl Drop for Shared {
    fn drop(&mut self) {
        if !EXCLUSIVE.get() {
            SHARED.set(SHARED.get() - 1);
        }
    }
}

impl Drop for Exclusive {
    fn drop(&mut self) {
        EXCLUSIVE.set(false);
    }
}
