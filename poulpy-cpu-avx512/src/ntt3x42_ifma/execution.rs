pub(crate) use poulpy_cpu_portable::kernels::SendPtr;

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
