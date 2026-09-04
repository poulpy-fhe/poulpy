//! Coefficient-range-parallel normalization: carries propagate across limbs
//! per coefficient, so disjoint coefficient ranges split freely and the carry
//! reservation is unchanged.

use poulpy_cpu_ref::reference::{
    ntt4x30::vec_znx_big::{I128NormalizeOps, ntt4x30_vec_znx_big_normalize, ntt4x30_vec_znx_big_normalize_range_raw},
    vec_znx::{vec_znx_normalize, vec_znx_normalize_assign, vec_znx_normalize_assign_range_raw, vec_znx_normalize_range_raw},
    znx::{
        ZnxAddAssign, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
    },
};
use poulpy_hal::layouts::{ArithmeticState, Backend, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendRef};
use rayon::prelude::*;

use crate::{RayonTaskExecutor, RayonTuning, SendPtr};
use poulpy_hal::layouts::{DataView, DataViewMut};

/// Coefficient-range tasks for one normalize over `n` coefficients.
pub fn normalize_tasks<B: RayonTuning>(n: usize) -> usize {
    if RayonTaskExecutor::should_serialize_inner() {
        return 1;
    }
    (n / B::NORMALIZE_MIN_TASK).clamp(1, ::rayon::current_num_threads())
}

/// Runs `run` in parallel over `tasks` coefficient ranges, each with its own
/// `words`-per-coefficient slice of `carry`.
fn for_each_range<T, F>(n: usize, tasks: usize, words: usize, carry: &mut [T], run: F)
where
    T: Send,
    F: Fn(usize, usize, &mut [T]) + Send + Sync,
{
    let chunk = n.div_ceil(tasks).next_multiple_of(8);
    carry[..words * n]
        .par_chunks_mut(words * chunk)
        .enumerate()
        .for_each(|(t, task_carry)| {
            let start = t * chunk;
            run(start, chunk.min(n - start), task_carry);
        });
}

/// Parallel [`vec_znx_normalize`], `B` being the serial kernel backend.
#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize_par<B, T>(
    res: &mut VecZnxBackendMut<'_, B, impl ArithmeticState>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'_, B, impl ArithmeticState>,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    B: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        return vec_znx_normalize::<B>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    let (cols, size) = (res.cols(), res.size());
    let (a_cols, a_size) = (a.cols(), a.size());
    let res_ptr = SendPtr::new(res.data_mut().as_mut_ptr().cast::<i64>());
    let a_data: &[u8] = a.data();
    for_each_range(n, tasks, 3, carry, |start, len, task_carry| {
        let a_view: VecZnxBackendRef<'_, B> = VecZnx::from_data(a_data, n, a_cols, a_size);
        unsafe {
            vec_znx_normalize_range_raw::<B>(
                res_ptr.get(),
                n,
                cols,
                size,
                res_base2k,
                res_offset,
                res_col,
                &a_view,
                a_base2k,
                a_col,
                start,
                len,
                task_carry,
            )
        }
    });
}

/// Parallel [`vec_znx_normalize_assign`], `B` being the serial kernel backend.
pub fn vec_znx_normalize_assign_par<B, T>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'_, B, impl ArithmeticState>,
    res_col: usize,
    carry: &mut [i64],
) where
    B: Backend<ZnxWord = i64> + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        return vec_znx_normalize_assign::<B>(base2k, res, res_col, carry);
    }

    let (cols, size) = (res.cols(), res.size());
    let res_ptr = SendPtr::new(res.data_mut().as_mut_ptr().cast::<i64>());
    for_each_range(n, tasks, 1, carry, |start, len, task_carry| unsafe {
        vec_znx_normalize_assign_range_raw::<B>(res_ptr.get(), n, cols, size, base2k, res_col, start, len, task_carry)
    });
}

/// Parallel [`ntt4x30_vec_znx_big_normalize`], `B` being the serial kernel
/// backend.
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_vec_znx_big_normalize_par<B, T>(
    res: &mut VecZnxBackendMut<'_, B, impl ArithmeticState>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, B>,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    B: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    B: 'static,
    T: RayonTuning,
{
    let n = res.n();
    let tasks = normalize_tasks::<T>(n);
    if tasks < 2 {
        let mut res_ref: &mut VecZnxBackendMut<'_, B, _> = res;
        let a_ref: &VecZnxBigBackendRef<'_, B> = a;
        return ntt4x30_vec_znx_big_normalize::<_, _, B>(
            &mut res_ref,
            res_base2k,
            res_offset,
            res_col,
            &a_ref,
            a_base2k,
            a_col,
            carry,
        );
    }

    let (cols, size) = (res.cols(), res.size());
    let (a_cols, a_size) = (a.cols(), a.size());
    let res_ptr = SendPtr::new(res.data_mut().as_mut_ptr().cast::<i64>());
    let a_data: &[u8] = a.data();
    for_each_range(n, tasks, 3, carry, |start, len, task_carry| {
        let a_view: VecZnxBigBackendRef<'_, B> = VecZnxBig::from_data(a_data, n, a_cols, a_size);
        unsafe {
            ntt4x30_vec_znx_big_normalize_range_raw::<_, B>(
                res_ptr.get(),
                n,
                cols,
                size,
                res_base2k,
                res_offset,
                res_col,
                &&a_view,
                a_base2k,
                a_col,
                start,
                len,
                task_carry,
            )
        }
    });
}
