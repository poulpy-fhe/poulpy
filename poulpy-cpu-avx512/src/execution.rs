//! Optional task executor owned by the AVX-512 backend crate.

use poulpy_hal::execution::TaskExecutor;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RayonTaskExecutor;

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
        ::rayon::join(left, right)
    }

    fn for_each_init<S, I, F>(count: usize, init: I, task: F)
    where
        S: Send,
        I: Fn() -> S + Send + Sync,
        F: Fn(&mut S, usize) + Send + Sync,
    {
        use ::rayon::prelude::*;
        (0..count).into_par_iter().for_each_init(init, task);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use poulpy_hal::execution::TaskExecutor;

    use super::RayonTaskExecutor;

    #[test]
    fn join_runs_both_tasks() {
        let pool = ::rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        pool.install(|| {
            let left = Arc::clone(&barrier);
            let right = Arc::clone(&barrier);
            <RayonTaskExecutor as TaskExecutor>::join(
                || {
                    left.wait();
                },
                || {
                    right.wait();
                },
            );
            assert_eq!(<RayonTaskExecutor as TaskExecutor>::max_parallelism(), 2);
        });
    }
}
