//! Optional task executor owned by the ARM backend crate.

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
    pub(crate) fn should_serialize_inner() -> bool {
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

    fn for_each_init<S, I, F>(count: usize, init: I, task: F)
    where
        S: Send,
        I: Fn() -> S + Send + Sync,
        F: Fn(&mut S, usize) + Send + Sync,
    {
        use ::rayon::prelude::*;
        if count == 0 {
            return;
        }
        let min_len = count.div_ceil(::rayon::current_num_threads().max(1));
        let depth = TASK_DEPTH.get() + 1;
        (0..count)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each_init(init, |state, index| {
                let _guard = TaskGuard::enter(depth);
                task(state, index);
            });
    }

    fn for_each_init_with_parallelism<S, I, F>(count: usize, parallelism: usize, init: I, task: F)
    where
        S: Send,
        I: Fn() -> S + Send + Sync,
        F: Fn(&mut S, usize) + Send + Sync,
    {
        use ::rayon::prelude::*;
        if count == 0 {
            return;
        }
        let workers = parallelism.min(::rayon::current_num_threads()).max(1);
        let min_len = count.div_ceil(workers);
        let depth = TASK_DEPTH.get() + 1;
        (0..count)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each_init(init, |state, index| {
                let _guard = TaskGuard::enter(depth);
                task(state, index);
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
