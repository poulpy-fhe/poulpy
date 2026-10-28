//! A counting global allocator: the system allocator, counting the bytes it holds and their high-water mark.
//!
//! A binary opts in with `#[global_allocator] static HEAP: poulpy_bench::heap::Counting = poulpy_bench::heap::Counting;`.
//! Without it the counters stay at zero, which [`counting`] reports.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

/// The system allocator, counting into [`live`] and [`peak`].
pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, size) };
        if !q.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            grow(size);
        }
        q
    }
}

/// Bytes the allocator holds.
pub fn live() -> usize {
    LIVE.load(Relaxed)
}

/// The most bytes the allocator held since the last [`reset_peak`].
pub fn peak() -> usize {
    PEAK.load(Relaxed)
}

/// Lowers the high-water mark to the bytes held now, and returns them.
pub fn reset_peak() -> usize {
    let live = live();
    PEAK.store(live, Relaxed);
    live
}

/// Whether [`Counting`] is the global allocator, from whether an allocation moves [`live`].
pub fn counting() -> bool {
    let before = live();
    let probe = black_box(Box::new([0u8; 64]));
    let moved = live() != before;
    drop(probe);
    moved
}
