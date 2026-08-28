//! Backend-selected task execution.

use crate::layouts::{Backend, ScratchArena};

pub trait TaskExecutor: Send + Sync + 'static {
    const IS_PARALLEL: bool;

    fn is_parallel() -> bool {
        Self::IS_PARALLEL
    }

    /// Tasks this executor can run concurrently: the pool width, or one when serial.
    fn max_parallelism() -> usize {
        1
    }

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;

    fn for_each<F>(count: usize, task: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        for index in 0..count {
            task(index);
        }
    }

    /// Applies `task` to every index, over `per_worker`-sized slices of `scratch`.
    fn for_each_chunked<T, F>(count: usize, scratch: &mut [T], per_worker: usize, task: F)
    where
        T: Send,
        F: Fn(&mut [T], usize) + Send + Sync,
    {
        assert!(
            scratch.len() >= per_worker,
            "worker scratch: {} < per_worker {per_worker}",
            scratch.len()
        );
        let worker = &mut scratch[..per_worker];
        for index in 0..count {
            task(worker, index);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SerialTaskExecutor;

impl TaskExecutor for SerialTaskExecutor {
    const IS_PARALLEL: bool = false;

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (left(), right())
    }
}

/// Number of concurrent workers for `tasks` independent tasks.
pub fn worker_count<E: TaskExecutor>(requested: usize, tasks: usize) -> usize {
    assert!(requested > 0, "parallel worker count must be non-zero");
    if tasks == 0 {
        0
    } else if E::is_parallel() {
        requested.min(E::max_parallelism()).min(tasks).max(1)
    } else {
        1
    }
}

/// Worker slices each kernel family is sized for, declared per backend.
///
/// The defaults are serial; a backend overrides the families it has measured.
pub trait ScratchWorkers {
    /// `cnv_prepare_left`, `cnv_prepare_right`, `cnv_prepare_self`.
    const PREPARE: usize = 1;
    /// `cnv_apply_dft`, `cnv_pairwise_apply_dft`, `cnv_accumulate_dft`.
    const APPLY: usize = 1;
    /// `vmp_apply_dft_to_dft` and its accumulate and strided variants.
    const VMP: usize = 1;
    /// `vec_znx_idft_apply`.
    const IDFT: usize = 1;
}

/// Worker slices to size scratch for, given the kernel's [`ScratchWorkers`] cap
/// min'd with the task count where that is known.
///
/// Independent of the pool width, so a reservation cannot change between the
/// call that sizes it and the call that uses it.
pub fn scratch_workers<E: TaskExecutor>(cap: usize) -> usize {
    if E::IS_PARALLEL { cap.max(1) } else { 1 }
}

/// [`scratch_workers`], bounded by the scratch bytes actually available.
///
/// Returns at least one: one worker's scratch is a precondition of every kernel,
/// and the caller's take reports the shortfall when it does not fit.
pub fn scratch_workers_within<E: TaskExecutor>(cap: usize, per_worker: usize, available: usize) -> usize {
    if per_worker == 0 {
        return 1;
    }
    scratch_workers::<E>(cap).min(available / per_worker).max(1)
}

/// Per-worker scratch size, aligned to the backend's scratch alignment.
pub fn worker_scratch_bytes<B: Backend>(bytes: usize) -> usize {
    assert!(B::SCRATCH_ALIGN > 0, "backend scratch alignment must be non-zero");
    bytes.next_multiple_of(B::SCRATCH_ALIGN)
}

/// Applies `task` to every item, distributing them over the given arenas.
pub fn for_each_with_scratch<E, B, T, F>(items: &mut [T], base_index: usize, scratch: Vec<ScratchArena<'_, B>>, task: &F)
where
    E: TaskExecutor,
    B: Backend,
    T: Send,
    F: Fn(usize, &mut T, &mut ScratchArena<'_, B>) + Send + Sync,
{
    assert!(!items.is_empty());
    assert!(!scratch.is_empty());
    assert!(scratch.len() <= items.len());

    if scratch.len() == 1 {
        let mut scratch = scratch.into_iter().next().unwrap();
        for (offset, item) in items.iter_mut().enumerate() {
            task(base_index + offset, item, &mut scratch);
        }
        return;
    }

    let left_workers = scratch.len() / 2;
    let split = items.len() * left_workers / scratch.len();
    let (left_items, right_items) = items.split_at_mut(split);
    let mut left_scratch = scratch;
    let right_scratch = left_scratch.split_off(left_workers);

    E::join(
        || for_each_with_scratch::<E, B, T, F>(left_items, base_index, left_scratch, task),
        || for_each_with_scratch::<E, B, T, F>(right_items, base_index + split, right_scratch, task),
    );
}

#[cfg(test)]
mod tests {
    use super::{SerialTaskExecutor, TaskExecutor, scratch_workers, scratch_workers_within, worker_count};

    #[test]
    fn serial_executor_always_uses_one_worker() {
        assert!(!SerialTaskExecutor::is_parallel());
        assert_eq!(worker_count::<SerialTaskExecutor>(8, 32), 1);
        assert_eq!(worker_count::<SerialTaskExecutor>(8, 0), 0);
        assert_eq!(scratch_workers::<SerialTaskExecutor>(32), 1);
        assert_eq!(scratch_workers_within::<SerialTaskExecutor>(32, 64, 4096), 1);
    }

    #[test]
    fn scratch_workers_are_bounded_by_the_arena() {
        assert_eq!(scratch_workers_within::<SerialTaskExecutor>(32, 64, 4096), 1);
        assert_eq!(scratch_workers_within::<SerialTaskExecutor>(32, 64, 0), 1);
        assert_eq!(scratch_workers_within::<SerialTaskExecutor>(32, 0, 4096), 1);
    }

    #[test]
    fn serial_for_each_chunked_visits_every_index_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut scratch = [0usize; 4];
        let visits: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
        SerialTaskExecutor::for_each_chunked(7, &mut scratch, 4, |worker, i| {
            assert_eq!(worker.len(), 4);
            visits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(visits.iter().all(|v| v.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn for_each_chunked_accepts_zero_tasks() {
        let mut scratch = [0usize; 4];
        SerialTaskExecutor::for_each_chunked(0, &mut scratch, 4, |_, _| panic!("no task expected"));
    }

    #[test]
    #[should_panic(expected = "worker scratch")]
    fn for_each_chunked_rejects_undersized_scratch() {
        let mut scratch = [0usize; 2];
        SerialTaskExecutor::for_each_chunked(1, &mut scratch, 4, |_, _| {});
    }
}
