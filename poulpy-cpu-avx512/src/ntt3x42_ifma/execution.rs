#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

// Dereferencing remains unsafe; users must enforce the pointee's aliasing rules.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

#[inline(always)]
pub(crate) fn for_index_with<S>(count: usize, _work: usize, init: impl FnOnce() -> S, mut task: impl FnMut(&mut S, usize)) {
    let mut state = init();
    for index in 0..count {
        task(&mut state, index);
    }
}

#[inline(always)]
pub(crate) fn for_index_exec<E: poulpy_hal::execution::TaskExecutor>(
    count: usize,
    work: usize,
    task: impl Fn(usize) + Send + Sync,
) {
    const PAR_MIN_WORK: usize = 1 << 17;
    if E::is_parallel() && count > 1 && work >= PAR_MIN_WORK {
        E::for_each(count, task);
    } else {
        for index in 0..count {
            task(index);
        }
    }
}
