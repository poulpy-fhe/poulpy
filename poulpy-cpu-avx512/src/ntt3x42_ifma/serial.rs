//! Serial task dispatch used by the single-threaded IFMA kernels.

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
