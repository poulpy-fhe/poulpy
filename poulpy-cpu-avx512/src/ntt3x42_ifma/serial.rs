//! Serial task dispatch used by the single-threaded IFMA kernels.

#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

impl<T> SendPtr<T> {
    #[inline(always)]
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

#[inline(always)]
pub(crate) fn for_index(count: usize, _work: usize, mut task: impl FnMut(usize)) {
    for index in 0..count {
        task(index);
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
pub(crate) const fn use_task_split(_count: usize, _work: usize) -> bool {
    false
}
