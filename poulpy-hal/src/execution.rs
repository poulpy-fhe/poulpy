//! Backend-selected task execution.

pub trait TaskExecutor: Send + Sync + 'static {
    const IS_PARALLEL: bool;

    fn is_parallel() -> bool {
        Self::IS_PARALLEL
    }

    /// Maximum number of tasks that should be scheduled concurrently by a
    /// backend-generic composite operation.
    ///
    /// Executors backed by a dynamic pool should report the pool width at the
    /// call site. The default keeps runtime-neutral and serial executors at one.
    fn max_parallelism() -> usize {
        1
    }

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;

    fn for_each_init<S, I, F>(count: usize, init: I, task: F)
    where
        S: Send,
        I: Fn() -> S + Send + Sync,
        F: Fn(&mut S, usize) + Send + Sync,
    {
        let mut state = init();
        for index in 0..count {
            task(&mut state, index);
        }
    }

    fn for_each_init_with_parallelism<S, I, F>(count: usize, _parallelism: usize, init: I, task: F)
    where
        S: Send,
        I: Fn() -> S + Send + Sync,
        F: Fn(&mut S, usize) + Send + Sync,
    {
        Self::for_each_init(count, init, task);
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
