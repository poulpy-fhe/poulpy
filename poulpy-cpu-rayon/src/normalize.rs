//! Coefficient-range-parallel normalization: carries propagate across limbs
//! per coefficient, so disjoint coefficient ranges split freely.

use poulpy_cpu_portable::kernels::{
    vec_znx::{
        vec_znx_normalize_assign_portable, vec_znx_normalize_assign_range_raw_portable, vec_znx_normalize_portable,
        vec_znx_normalize_range_raw_portable,
    },
    vec_znx_big_normalize::{vec_znx_big_normalize_portable, vec_znx_big_normalize_range_raw_portable},
    znx_normalize::ZnxNormalize,
};
use poulpy_hal::layouts::{
    Backend, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendRef, ZnxView, ZnxViewMut,
};
use rayon::prelude::*;

use crate::{RayonTaskExecutor, RayonTuning, SendPtr};

/// Coefficient-range tasks for one normalize over `n` coefficients.
pub fn normalize_tasks<B: RayonTuning>(n: usize) -> usize {
    if RayonTaskExecutor::should_serialize_inner() {
        return 1;
    }
    (n / B::NORMALIZE_MIN_TASK).clamp(1, ::rayon::current_num_threads())
}

/// Runs `run` in parallel over `tasks` coefficient ranges, each starting at a multiple of 8.
///
/// `n` is the visible degree of the destination, so ranges index a window, not
/// the buffer. The split path (`tasks >= 2`) is pinned on windows by the
/// `window_rayon` suites that run at degree `1 << 13` (`poulpy-cpu-avx` NTT4x30,
/// `poulpy-cpu-avx512` FFT64); suites at `1 << 8` only reach the serial fallback.
fn for_each_range<F>(n: usize, tasks: usize, run: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    if n == 0 {
        return;
    }
    let chunk = n.div_ceil(tasks).next_multiple_of(8);
    (0..n.div_ceil(chunk)).into_par_iter().for_each(|t| {
        let start = t * chunk;
        run(start, chunk.min(n - start));
    });
}

/// Parallel [`vec_znx_normalize`], `B` being the serial kernel backend.
#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize_par<B, T>(
    res: &mut VecZnxBackendMut<'_, B>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'_, B>,
    a_base2k: usize,
    a_col: usize,
) where
    B: ZnxNormalize,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        return vec_znx_normalize_portable::<B>(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col);
    }

    assert!(res_col < res.cols(), "operand column out of range");
    if res.size() > 0 {
        let _ = res.at_ptr(res_col, res.size() - 1);
    }
    let res_shape = res.shape();
    let a_shape = a.shape();
    let res_ptr = SendPtr::new(res.base_mut_ptr());
    let a_data: &[u8] = a.data();
    for_each_range(n, tasks, |start, len| {
        let a_view: VecZnxBackendRef<'_, B> = VecZnx::from_shape(a_data, a_shape);
        // SAFETY: `res` is bounds-checked above and exclusive; tasks write disjoint coefficient ranges.
        unsafe {
            vec_znx_normalize_range_raw_portable::<B>(
                res_ptr.get(),
                res_shape,
                res_base2k,
                res_k,
                res_offset,
                res_col,
                &a_view,
                a_base2k,
                a_col,
                start,
                len,
            )
        }
    });
}

/// Parallel [`vec_znx_normalize_assign`], `B` being the serial kernel backend.
pub fn vec_znx_normalize_assign_par<B, T>(
    base2k: usize,
    k: usize,
    res_offset: i64,
    res: &mut VecZnxBackendMut<'_, B>,
    res_col: usize,
) where
    B: ZnxNormalize,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        return vec_znx_normalize_assign_portable::<B>(base2k, k, res_offset, res, res_col);
    }

    assert!(res_col < res.cols(), "operand column out of range");
    if res.size() > 0 {
        let _ = res.at_ptr(res_col, res.size() - 1);
    }
    let res_shape = res.shape();
    let res_ptr = SendPtr::new(res.base_mut_ptr());
    // SAFETY: `res` is bounds-checked above and exclusive; tasks write disjoint coefficient ranges.
    for_each_range(n, tasks, |start, len| unsafe {
        vec_znx_normalize_assign_range_raw_portable::<B>(res_ptr.get(), res_shape, base2k, k, res_offset, res_col, start, len)
    });
}

/// Parallel [`vec_znx_big_normalize`], `B` being the serial kernel backend.
#[allow(clippy::too_many_arguments)]
pub fn vec_znx_big_normalize_par<B, T>(
    res: &mut VecZnxBackendMut<'_, B>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, B>,
    a_base2k: usize,
    a_col: usize,
) where
    B: ZnxNormalize,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        return vec_znx_big_normalize_portable::<B>(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col);
    }

    assert!(res_col < res.cols(), "operand column out of range");
    if res.size() > 0 {
        let _ = res.at_ptr(res_col, res.size() - 1);
    }
    let res_shape = res.shape();
    let a_shape = a.shape();
    let res_ptr = SendPtr::new(res.base_mut_ptr());
    let a_data: &[u8] = a.data;
    for_each_range(n, tasks, |start, len| {
        let a_view: VecZnxBigBackendRef<'_, B> = VecZnxBig::from_shape(a_data, a_shape);
        // SAFETY: `res` is bounds-checked above and exclusive; tasks write disjoint coefficient ranges.
        unsafe {
            vec_znx_big_normalize_range_raw_portable::<B>(
                res_ptr.get(),
                res_shape,
                res_base2k,
                res_k,
                res_offset,
                res_col,
                &a_view,
                a_base2k,
                a_col,
                start,
                len,
            )
        }
    });
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::for_each_range;

    #[test]
    fn test_normalization_ranges_cover_once() {
        rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
            for n in 0..=129 {
                for tasks in 1..=17 {
                    let visits: Vec<_> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                    for_each_range(n, tasks, |start, len| {
                        assert!(len > 0 && start + len <= n && start % 8 == 0);
                        for visit in &visits[start..start + len] {
                            assert_eq!(visit.fetch_add(1, Ordering::Relaxed), 0);
                        }
                    });
                    assert!(visits.iter().all(|v| v.load(Ordering::Relaxed) == 1));
                }
            }
        });
    }
}
