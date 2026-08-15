//! Backend-selected parallel work partitioning.
//!
//! This module deliberately depends only on the HAL executor abstraction. The
//! concrete thread-pool implementation remains owned by the selected backend.

use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{Backend, ScratchArena},
};

pub(crate) fn worker_count<E: TaskExecutor>(requested: usize, tasks: usize) -> usize {
    assert!(requested > 0, "parallel worker count must be non-zero");
    if tasks == 0 {
        0
    } else if E::is_parallel() {
        requested.min(E::max_parallelism()).min(tasks).max(1)
    } else {
        1
    }
}

pub(crate) fn worker_scratch_bytes<BE: Backend>(bytes: usize) -> usize {
    let align = BE::SCRATCH_ALIGN;
    assert!(align > 0, "backend scratch alignment must be non-zero");
    bytes
        .checked_add(align - 1)
        .expect("worker scratch alignment overflows usize")
        / align
        * align
}

pub(crate) fn for_each_with_scratch<E, BE, T, F>(items: &mut [T], base_index: usize, scratch: Vec<ScratchArena<'_, BE>>, task: &F)
where
    E: TaskExecutor,
    BE: Backend,
    T: Send,
    F: Fn(usize, &mut T, &mut ScratchArena<'_, BE>) + Send + Sync,
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
        || for_each_with_scratch::<E, BE, T, F>(left_items, base_index, left_scratch, task),
        || for_each_with_scratch::<E, BE, T, F>(right_items, base_index + split, right_scratch, task),
    );
}

#[cfg(test)]
mod tests {
    use poulpy_hal::execution::{SerialTaskExecutor, TaskExecutor};

    use super::worker_count;

    #[test]
    fn serial_executor_always_uses_one_worker() {
        assert!(!SerialTaskExecutor::is_parallel());
        assert_eq!(worker_count::<SerialTaskExecutor>(8, 32), 1);
        assert_eq!(worker_count::<SerialTaskExecutor>(8, 0), 0);
    }
}
