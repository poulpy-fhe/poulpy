use poulpy_cpu_rayon::RayonTaskExecutor;
use poulpy_hal::execution::TaskExecutor;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// IFMA executor that confines independent branches to separate shared caches.
pub struct NTT3x42IfmaRayonExecutor;

impl NTT3x42IfmaRayonExecutor {
    pub fn should_serialize_inner() -> bool {
        RayonTaskExecutor::should_serialize_inner()
    }
}

impl TaskExecutor for NTT3x42IfmaRayonExecutor {
    const IS_PARALLEL: bool = true;

    fn is_parallel() -> bool {
        RayonTaskExecutor::is_parallel()
    }

    fn max_parallelism() -> usize {
        RayonTaskExecutor::max_parallelism()
    }

    fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        if let Some(pools) = CachePools::new() {
            pools.join(left, right)
        } else {
            RayonTaskExecutor::join(left, right)
        }
    }

    fn for_each<F: Fn(usize) + Send + Sync>(count: usize, task: F) {
        RayonTaskExecutor::for_each(count, task);
    }

    fn for_each_chunked<T: Send, F: Fn(&mut [T], usize) + Send + Sync>(
        count: usize,
        scratch: &mut [T],
        per_worker: usize,
        task: F,
    ) {
        RayonTaskExecutor::for_each_chunked(count, scratch, per_worker, task);
    }
}

pub(crate) struct CachePools([rayon::ThreadPool; 2]);

impl CachePools {
    pub(crate) fn new() -> Option<Self> {
        if RayonTaskExecutor::should_serialize_inner() {
            return None;
        }
        #[cfg(target_os = "linux")]
        {
            let groups = cache_groups(rayon::current_num_threads())?;
            let mut pools = Vec::with_capacity(2);
            for cpus in groups {
                let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let errors = failed.clone();
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(cpus.len())
                    .start_handler(move |worker| unsafe {
                        let mut mask: libc::cpu_set_t = std::mem::zeroed();
                        libc::CPU_ZERO(&mut mask);
                        libc::CPU_SET(cpus[worker], &mut mask);
                        if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &mask) != 0 {
                            errors.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    })
                    .build()
                    .ok()?;
                pool.broadcast(|_| ());
                if failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return None;
                }
                pools.push(pool);
            }
            Some(Self(pools.try_into().ok()?))
        }
        #[cfg(not(target_os = "linux"))]
        None
    }

    pub(crate) fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        std::thread::scope(|scope| {
            let left = scope.spawn(|| self.0[0].install(|| RayonTaskExecutor::join(left, || ()).0));
            let right = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.0[1].install(|| RayonTaskExecutor::join(right, || ()).0)
            }));
            match (left.join(), right) {
                (Ok(left), Ok(right)) => (left, right),
                (Err(panic), _) | (_, Err(panic)) => std::panic::resume_unwind(panic),
            }
        })
    }
}

#[cfg(target_os = "linux")]
fn cache_groups(threads: usize) -> Option<[Vec<usize>; 2]> {
    use std::collections::BTreeMap;
    if threads < 4 || !threads.is_multiple_of(2) {
        return None;
    }
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut mask) } != 0 {
        return None;
    }
    let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if !unsafe { libc::CPU_ISSET(cpu, &mask) } {
            continue;
        }
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
        let mut last = None;
        for entry in std::fs::read_dir(path).ok()? {
            let path = entry.ok()?.path();
            if !path.is_dir() {
                continue;
            }
            let level = std::fs::read_to_string(path.join("level"))
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()?;
            if last.as_ref().is_none_or(|(old, _)| level > *old) {
                let cpus = parse_cpus(&std::fs::read_to_string(path.join("shared_cpu_list")).ok()?)?;
                last = Some((level, cpus));
            }
        }
        groups.entry(last?.1).or_default().push(cpu);
    }
    if groups.len() != 2 {
        return None;
    }
    let mut groups = groups.into_values();
    let mut result = [groups.next()?, groups.next()?];
    for group in &mut result {
        if group.len() < threads / 2 {
            return None;
        }
        group.truncate(threads / 2);
    }
    Some(result)
}

#[cfg(target_os = "linux")]
fn parse_cpus(value: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for item in value.trim().split(',') {
        if let Some((first, last)) = item.split_once('-') {
            let first = first.parse::<usize>().ok()?;
            let last = last.parse::<usize>().ok()?;
            if first > last || last >= libc::CPU_SETSIZE as usize {
                return None;
            }
            cpus.extend(first..=last);
        } else {
            let cpu = item.parse::<usize>().ok()?;
            if cpu >= libc::CPU_SETSIZE as usize {
                return None;
            }
            cpus.push(cpu);
        }
    }
    Some(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_branches_finish_before_a_panic_is_propagated() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pools = CachePools(std::array::from_fn(|_| {
            rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap()
        }));
        let finished = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pools.join(
                || {
                    finished.fetch_add(1, Ordering::SeqCst);
                    panic!("left");
                },
                || {
                    finished.fetch_add(1, Ordering::SeqCst);
                    panic!("right");
                },
            )
        }));
        assert!(result.is_err());
        assert_eq!(finished.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn join_restores_nesting_and_borrows_both_results() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let mut values = [0, 0];
        let (left, right) = values.split_at_mut(1);
        pool.install(|| {
            NTT3x42IfmaRayonExecutor::join(
                || {
                    left[0] = 3;
                    assert!(!NTT3x42IfmaRayonExecutor::should_serialize_inner());
                },
                || {
                    right[0] = 5;
                    assert_eq!(rayon::current_num_threads(), 2);
                },
            );
            assert!(!NTT3x42IfmaRayonExecutor::should_serialize_inner());
        });
        assert_eq!(values, [3, 5]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_lists_reject_invalid_ranges() {
        assert_eq!(parse_cpus("0-3,8,12-13\n"), Some(vec![0, 1, 2, 3, 8, 12, 13]));
        for invalid in ["", "3-1", "abc", "0-1024", "1024"] {
            assert_eq!(parse_cpus(invalid), None);
        }
    }
}
