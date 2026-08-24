//! Rayon-scheduled task executor and backend wrappers shared by the CPU backend crates.

pub mod fft64;
pub mod normalize;
pub mod tuning;

/// Re-exports for the crate's macros. Not a stable API.
#[doc(hidden)]
pub mod __private {
    pub use ::bytemuck;
    pub use ::poulpy_cpu_ref;
    pub use ::poulpy_hal;
    pub use ::rayon;
}

use std::cell::Cell;

use poulpy_hal::execution::TaskExecutor;

thread_local! {
    static TASK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct TaskGuard(usize);

impl TaskGuard {
    fn enter(depth: usize) -> Self {
        Self(TASK_DEPTH.replace(depth))
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        TASK_DEPTH.set(self.0);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RayonTaskExecutor;

impl RayonTaskExecutor {
    pub fn should_serialize_inner() -> bool {
        TASK_DEPTH.get() > 1
    }
}

impl TaskExecutor for RayonTaskExecutor {
    const IS_PARALLEL: bool = true;

    fn is_parallel() -> bool {
        ::rayon::current_num_threads() > 1
    }

    fn max_parallelism() -> usize {
        ::rayon::current_num_threads().max(1)
    }

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let depth = TASK_DEPTH.get() + 1;
        ::rayon::join(
            || {
                let _guard = TaskGuard::enter(depth);
                left()
            },
            || {
                let _guard = TaskGuard::enter(depth);
                right()
            },
        )
    }

    fn for_each<F>(count: usize, task: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        use ::rayon::prelude::*;
        if count == 0 {
            return;
        }
        let min_len = count.div_ceil(::rayon::current_num_threads().max(1));
        let depth = TASK_DEPTH.get() + 1;
        (0..count).into_par_iter().with_min_len(min_len).for_each(|index| {
            let _guard = TaskGuard::enter(depth);
            task(index);
        });
    }

    fn for_each_chunked<T, F>(count: usize, scratch: &mut [T], per_worker: usize, task: F)
    where
        T: Send,
        F: Fn(&mut [T], usize) + Send + Sync,
    {
        use ::rayon::prelude::*;
        assert!(
            scratch.len() >= per_worker,
            "worker scratch: {} < per_worker {per_worker}",
            scratch.len()
        );
        if count == 0 {
            return;
        }
        let workers = scratch
            .len()
            .checked_div(per_worker)
            .unwrap_or(1)
            .min(count)
            .min(::rayon::current_num_threads())
            .max(1);
        if workers == 1 {
            let worker = &mut scratch[..per_worker];
            for index in 0..count {
                task(worker, index);
            }
            return;
        }
        let span = count.div_ceil(workers);
        let depth = TASK_DEPTH.get() + 1;
        scratch[..workers * per_worker]
            .par_chunks_mut(per_worker)
            .enumerate()
            .for_each(|(worker, buffer)| {
                let _guard = TaskGuard::enter(depth);
                let start = worker * span;
                for index in start..(start + span).min(count) {
                    task(buffer, index);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use poulpy_hal::execution::TaskExecutor;

    use super::RayonTaskExecutor;

    #[test]
    fn join_uses_the_active_rayon_pool() {
        let pool = ::rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        pool.install(|| {
            assert!(!RayonTaskExecutor::should_serialize_inner());
            let left = Arc::clone(&barrier);
            let right = Arc::clone(&barrier);
            <RayonTaskExecutor as TaskExecutor>::join(
                || {
                    assert!(!RayonTaskExecutor::should_serialize_inner());
                    left.wait();
                },
                || {
                    assert!(!RayonTaskExecutor::should_serialize_inner());
                    right.wait();
                },
            );
            assert!(!RayonTaskExecutor::should_serialize_inner());
            assert_eq!(<RayonTaskExecutor as TaskExecutor>::max_parallelism(), 2);

            <RayonTaskExecutor as TaskExecutor>::join(
                || {
                    <RayonTaskExecutor as TaskExecutor>::join(
                        || assert!(RayonTaskExecutor::should_serialize_inner()),
                        || assert!(RayonTaskExecutor::should_serialize_inner()),
                    );
                },
                || {},
            );
        });
    }
}

/// Takes a `len`-element typed slice from the arena.
pub fn take_scratch<'a, BE, T>(
    arena: poulpy_hal::layouts::ScratchArena<'a, BE>,
    len: usize,
) -> (&'a mut [T], poulpy_hal::layouts::ScratchArena<'a, BE>)
where
    BE: poulpy_hal::layouts::Backend + 'a,
    BE::BufMut<'a>: poulpy_hal::api::HostBufMut<'a>,
    T: bytemuck::Pod,
{
    use poulpy_hal::api::HostBufMut;
    assert!(BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()));
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!((bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()));
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

/// Worker slices to size scratch for, capped by the kernel's own bound.
pub fn workers(cap: usize) -> usize {
    poulpy_hal::execution::scratch_workers::<RayonTaskExecutor>(cap)
}

/// Same as [`workers`], bounded by the scratch bytes actually available.
pub fn workers_within(cap: usize, per_worker: usize, available: usize) -> usize {
    poulpy_hal::execution::scratch_workers_within::<RayonTaskExecutor>(cap, per_worker, available)
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*mut T);

// Dereferencing remains unsafe; users must enforce the pointee's aliasing rules.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    pub fn get(self) -> *mut T {
        self.0
    }
}

/// Backend-local thresholds for Rayon scheduling.
pub trait RayonTuning {
    const COEFF_MIN_LEN: usize;
    const COEFF_MIN_TASK: usize;
    const NORMALIZE_MIN_TASK: usize;
}

/// Chunk length for coefficient-wise parallel kernels, or `None` when the slice
/// is too short to repay scheduling.
pub fn parallel_chunk_len<B: RayonTuning>(len: usize) -> Option<usize> {
    if len < B::COEFF_MIN_LEN || ::rayon::current_num_threads() < 2 {
        return None;
    }
    let tasks = ::rayon::current_num_threads().min(len / B::COEFF_MIN_TASK).max(1);
    Some(len.div_ceil(tasks).next_multiple_of(64))
}

/// Whether a per-limb task split is worth scheduling.
#[inline]
pub fn parallel_limb_tasks(count: usize) -> bool {
    count > 1 && ::rayon::current_num_threads() > 1
}

#[cfg(test)]
mod chunked_tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use poulpy_hal::execution::{TaskExecutor, scratch_workers};

    use super::RayonTaskExecutor;

    /// Records, per worker slice, the indices that ran on it.
    fn run(threads: usize, count: usize, workers: usize, per_worker: usize) -> BTreeMap<usize, Vec<usize>> {
        let word = size_of::<usize>();
        let pool = ::rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut scratch = vec![0usize; workers * per_worker];
        let base = scratch.as_ptr() as usize;
        let seen: Mutex<BTreeMap<usize, Vec<usize>>> = Mutex::new(BTreeMap::new());
        pool.install(|| {
            RayonTaskExecutor::for_each_chunked(count, &mut scratch, per_worker, |worker, index| {
                assert_eq!(worker.len(), per_worker);
                worker[0] = index;
                // A racing worker would observe a different value here.
                std::hint::black_box(&worker);
                assert_eq!(worker[0], index);
                let offset = worker.as_ptr() as usize - base;
                assert_eq!(offset % (per_worker * word), 0);
                seen.lock().unwrap().entry(offset).or_default().push(index);
            });
        });
        seen.into_inner().unwrap()
    }

    const PER_WORKER: usize = 8;

    #[test]
    fn every_index_runs_once_on_a_disjoint_worker_slice() {
        let seen = run(4, 100, 4, PER_WORKER);
        let mut indices: Vec<usize> = seen.values().flatten().copied().collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..100).collect::<Vec<_>>());
        assert!(seen.len() > 1, "expected several worker slices, got {}", seen.len());
        let offsets: Vec<usize> = seen.keys().copied().collect();
        let stride = PER_WORKER * size_of::<usize>();
        for pair in offsets.windows(2) {
            assert!(pair[1] - pair[0] >= stride, "worker slices overlap");
        }
    }

    #[test]
    fn a_single_thread_pool_uses_one_worker_slice() {
        assert_eq!(run(1, 32, 4, 8).len(), 1);
    }

    #[test]
    fn worker_count_never_exceeds_the_scratch_it_was_given() {
        assert_eq!(run(8, 32, 2, 8).len(), 2);
        assert_eq!(run(8, 3, 8, 8).len(), 3);
    }

    #[test]
    fn zero_tasks_run_nothing() {
        assert!(run(4, 0, 4, 8).is_empty());
    }

    #[test]
    #[should_panic(expected = "worker scratch")]
    fn undersized_scratch_is_rejected() {
        let mut scratch = [0usize; 4];
        RayonTaskExecutor::for_each_chunked(1, &mut scratch, 8, |_, _| {});
    }

    #[test]
    fn scratch_sizing_is_the_callers_cap_and_pool_independent() {
        assert_eq!(scratch_workers::<RayonTaskExecutor>(4), 4);
        assert_eq!(scratch_workers::<RayonTaskExecutor>(0), 1);
        let pool = ::rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| assert_eq!(scratch_workers::<RayonTaskExecutor>(8), 8));
    }
}
