use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256,
};
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::execution::TaskExecutor;
#[cfg(feature = "enable-rayon")]
use poulpy_hal::layouts::CnvDftAccTerm;
use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CrtWord, HostDataMut, HostDataRef,
    Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
};
use std::mem::size_of;

use super::{
    arithmetic_avx::{BARRETT_MU, POW32, Q_VEC, cond_sub, reduce_b_to_canonical},
    mat_vec_avx::reduce_bbc,
    vec_znx_dft::pack_two_q120,
};
const GROUP: usize = 8;

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

#[inline(always)]
fn packed_row_offset(size: usize, limb: usize, group: usize) -> usize {
    (group * size + limb) * 4 * GROUP
}

#[inline(always)]
fn col_slice(raw: &[u32], n: usize, size: usize, col: usize) -> &[u32] {
    let stride = 4 * n * size;
    &raw[col * stride..(col + 1) * stride]
}

#[inline(always)]
fn col_slice_mut(raw: &mut [u32], n: usize, size: usize, col: usize) -> &mut [u32] {
    let stride = 4 * n * size;
    &mut raw[col * stride..(col + 1) * stride]
}

fn zero_res_limb<BE>(res: &mut VecZnxDftBackendMut<'_, BE>, col: usize, limb: usize)
where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    cast_slice_mut::<_, u32>(res.at_mut(col, limb)).fill(0);
}

#[inline(always)]
unsafe fn reduce_accum(meta: &BbcMeta<Primes30>, lo: __m256i, hi: __m256i) -> __m256i {
    unsafe {
        let x = reduce_bbc(
            lo,
            hi,
            _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64),
            meta.h,
            _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i),
            _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i),
        );
        reduce_b_to_canonical(
            x,
            _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i),
            _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i),
            _mm256_loadu_si256(POW32.as_ptr() as *const __m256i),
        )
    }
}

#[inline(always)]
unsafe fn accumulate_product(lo: &mut __m256i, hi: &mut __m256i, a: __m256i, b: __m256i) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let product = _mm256_mul_epu32(a, b);
        *lo = _mm256_add_epi64(*lo, _mm256_and_si256(product, mask32));
        *hi = _mm256_add_epi64(*hi, _mm256_srli_epi64::<32>(product));
    }
}

#[inline(always)]
unsafe fn load_coeff(row: *const u32, coeff: usize) -> __m256i {
    unsafe { _mm256_cvtepu32_epi64(core::arch::x86_64::_mm_loadu_si128(row.add(4 * coeff) as *const _)) }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn conv_group<const ACC: bool, const PAIRWISE: bool>(
    meta: &BbcMeta<Primes30>,
    res: SendPtr<u32>,
    res_col: usize,
    n: usize,
    res_cols: usize,
    min_size: usize,
    offset: usize,
    group: usize,
    a0: &[u32],
    a1: &[u32],
    a_size: usize,
    b0: &[u32],
    b1: &[u32],
    b_size: usize,
) {
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            for pair in 0..GROUP / 2 {
                let mut lo0 = _mm256_setzero_si256();
                let mut hi0 = _mm256_setzero_si256();
                let mut lo1 = _mm256_setzero_si256();
                let mut hi1 = _mm256_setzero_si256();
                for row in 0..j_max - j_min {
                    let ao = packed_row_offset(a_size, a_start + row, group);
                    let bo = packed_row_offset(b_size, b_start + row, group);
                    let mut av0 = load_coeff(a0.as_ptr().add(ao), 2 * pair);
                    let mut av1 = load_coeff(a0.as_ptr().add(ao), 2 * pair + 1);
                    let mut bv0 = load_coeff(b0.as_ptr().add(bo), 2 * pair);
                    let mut bv1 = load_coeff(b0.as_ptr().add(bo), 2 * pair + 1);
                    if PAIRWISE {
                        av0 = cond_sub(_mm256_add_epi64(av0, load_coeff(a1.as_ptr().add(ao), 2 * pair)), q);
                        av1 = cond_sub(_mm256_add_epi64(av1, load_coeff(a1.as_ptr().add(ao), 2 * pair + 1)), q);
                        bv0 = cond_sub(_mm256_add_epi64(bv0, load_coeff(b1.as_ptr().add(bo), 2 * pair)), q);
                        bv1 = cond_sub(_mm256_add_epi64(bv1, load_coeff(b1.as_ptr().add(bo), 2 * pair + 1)), q);
                    }
                    accumulate_product(&mut lo0, &mut hi0, av0, bv0);
                    accumulate_product(&mut lo1, &mut hi1, av1, bv1);
                }
                let value0 = reduce_accum(meta, lo0, hi0);
                let value1 = reduce_accum(meta, lo1, hi1);
                let dst = res.get().add((k * res_cols + res_col) * 4 * n + group * 4 * GROUP + pair * 8);
                if ACC {
                    let old = _mm256_loadu_si256(dst as *const __m256i);
                    let old0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(old));
                    let old1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(old));
                    _mm256_storeu_si256(
                        dst as *mut __m256i,
                        pack_two_q120(
                            cond_sub(_mm256_add_epi64(old0, value0), q),
                            cond_sub(_mm256_add_epi64(old1, value1), q),
                        ),
                    );
                } else {
                    _mm256_storeu_si256(dst as *mut __m256i, pack_two_q120(value0, value1));
                }
            }
        }
    }
}

pub(crate) fn cnv_prepare_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

#[target_feature(enable = "avx2")]
unsafe fn pack_prepared_limb(dst: &mut [u32], src: &[u64], n: usize, size: usize, limb: usize) {
    unsafe {
        let q = _mm256_loadu_si256(super::arithmetic_avx::Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(super::arithmetic_avx::BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(super::arithmetic_avx::POW32.as_ptr() as *const __m256i);
        for group in 0..n / GROUP {
            let dst_off = packed_row_offset(size, limb, group);
            for pair in 0..GROUP / 2 {
                let src_off = group * 4 * GROUP + pair * 8;
                let a = super::arithmetic_avx::reduce_b_to_canonical(
                    _mm256_loadu_si256(src.as_ptr().add(src_off) as *const __m256i),
                    q,
                    mu,
                    pow32,
                );
                let b = super::arithmetic_avx::reduce_b_to_canonical(
                    _mm256_loadu_si256(src.as_ptr().add(src_off + 4) as *const __m256i),
                    q,
                    mu,
                    pow32,
                );
                _mm256_storeu_si256(dst.as_mut_ptr().add(dst_off + pair * 8) as *mut __m256i, pack_two_q120(a, b));
            }
        }
    }
}

fn prepare<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: Option<&mut CnvPVecLBackendMut<'_, BE>>,
    right: Option<&mut CnvPVecRBackendMut<'_, BE>>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_ref::reference::ntt4x30::ntt::NttTable<Primes30>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    let (n, cols, size) = if let Some(res) = left.as_ref() {
        (res.n(), res.cols(), res.size())
    } else {
        let res = right.as_ref().unwrap();
        (res.n(), res.cols(), res.size())
    };
    let min_size = size.min(a.size());
    let mut left = left.map(|res| cast_slice_mut::<_, u32>(res.raw_mut()));
    let mut right = right.map(|res| cast_slice_mut::<_, u32>(res.raw_mut()));
    let task_count = cols * size;
    if E::is_parallel() && task_count > 1 {
        let stride = 4 * n * size;
        let left_ptr = left.as_deref_mut().map(|data| SendPtr(data.as_mut_ptr()));
        let right_ptr = right.as_deref_mut().map(|data| SendPtr(data.as_mut_ptr()));
        E::for_each_chunked(task_count, tmp, 4 * n, |tmp, task| {
            let col = task / size;
            let limb = task % size;
            let mut dst_l = left_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
            let mut dst_r = right_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
            if limb < min_size {
                if limb + 1 == min_size {
                    BE::ntt_from_znx64_masked(tmp, a.at(col, limb), mask);
                } else {
                    BE::ntt_from_znx64(tmp, a.at(col, limb));
                }
                BE::ntt_dft_execute(module.get_ntt_table(), tmp);
                if let Some(dst) = dst_l.as_deref_mut() {
                    unsafe { pack_prepared_limb(dst, tmp, n, size, limb) };
                }
                if let Some(dst) = dst_r.as_deref_mut() {
                    unsafe { pack_prepared_limb(dst, tmp, n, size, size - 1 - limb) };
                }
            } else {
                for group in 0..n / GROUP {
                    if let Some(dst) = dst_l.as_deref_mut() {
                        let off = packed_row_offset(size, limb, group);
                        dst[off..off + 4 * GROUP].fill(0);
                    }
                    if let Some(dst) = dst_r.as_deref_mut() {
                        let off = packed_row_offset(size, size - 1 - limb, group);
                        dst[off..off + 4 * GROUP].fill(0);
                    }
                }
            }
        });
        return;
    }

    let tmp = &mut tmp[..4 * n];
    for col in 0..cols {
        let mut dst_l = left.as_deref_mut().map(|data| col_slice_mut(data, n, size, col));
        let mut dst_r = right.as_deref_mut().map(|data| col_slice_mut(data, n, size, col));
        for limb in 0..min_size {
            if limb + 1 == min_size {
                BE::ntt_from_znx64_masked(tmp, a.at(col, limb), mask);
            } else {
                BE::ntt_from_znx64(tmp, a.at(col, limb));
            }
            BE::ntt_dft_execute(module.get_ntt_table(), tmp);
            if let Some(dst) = dst_l.as_deref_mut() {
                unsafe { pack_prepared_limb(dst, tmp, n, size, limb) };
            }
            if let Some(dst) = dst_r.as_deref_mut() {
                unsafe { pack_prepared_limb(dst, tmp, n, size, size - 1 - limb) };
            }
        }
        for limb in min_size..size {
            for group in 0..n / GROUP {
                if let Some(dst) = dst_l.as_deref_mut() {
                    let off = packed_row_offset(size, limb, group);
                    dst[off..off + 4 * GROUP].fill(0);
                }
                if let Some(dst) = dst_r.as_deref_mut() {
                    let off = packed_row_offset(size, size - 1 - limb, group);
                    dst[off..off + 4 * GROUP].fill(0);
                }
            }
        }
    }
}

pub(crate) fn cnv_prepare_left<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_ref::reference::ntt4x30::ntt::NttTable<Primes30>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, Some(res), None, a, mask, tmp);
}

pub(crate) fn cnv_prepare_right<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_ref::reference::ntt4x30::ntt::NttTable<Primes30>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, None, Some(res), a, mask, tmp);
}

pub(crate) fn cnv_prepare_self<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_ref::reference::ntt4x30::ntt::NttTable<Primes30>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, Some(left), Some(right), a, mask, tmp);
}

pub(crate) fn cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
unsafe fn apply<BE, E: TaskExecutor, const ACC: bool, const PAIRWISE: bool>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a0_col: usize,
    a1_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b0_col: usize,
    b1_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    let (n, res_size, a_size, b_size) = (res.n(), res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        if !ACC {
            for limb in 0..res_size {
                zero_res_limb(res, res_col, limb);
            }
        }
        return;
    }
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let a_raw: &[u32] = cast_slice(a.raw());
    let b_raw: &[u32] = cast_slice(b.raw());
    let a0 = col_slice(a_raw, n, a_size, a0_col);
    let a1 = col_slice(a_raw, n, a_size, a1_col);
    let b0 = col_slice(b_raw, n, b_size, b0_col);
    let b1 = col_slice(b_raw, n, b_size, b1_col);
    let res_cols = res.cols();
    let res_ptr = SendPtr(cast_slice_mut::<_, u32>(res.raw_mut()).as_mut_ptr());
    E::for_each(n / GROUP, |group| unsafe {
        conv_group::<ACC, PAIRWISE>(
            module.get_bbc_meta(),
            res_ptr,
            res_col,
            n,
            res_cols,
            min_size,
            offset,
            group,
            a0,
            a1,
            a_size,
            b0,
            b1,
            b_size,
        )
    });
    if !ACC {
        for limb in min_size..res_size {
            zero_res_limb(res, res_col, limb);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    unsafe { apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft_accumulate<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    unsafe { apply::<BE, E, true, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[cfg(feature = "enable-rayon")]
pub(crate) fn cnv_accumulate_dft_avx_tmp_bytes(_res_size: usize) -> usize {
    0
}

#[cfg(feature = "enable-rayon")]
pub(crate) unsafe fn cnv_accumulate_dft_avx<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[CnvDftAccTerm<'_, BE>],
    _tmp: &mut [u8],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    if terms.is_empty() {
        for limb in 0..res.size() {
            zero_res_limb(res, res_col, limb);
        }
        return;
    }

    for (index, term) in terms.iter().enumerate() {
        if index == 0 {
            unsafe {
                apply::<BE, E, false, false>(
                    module, cnv_offset, res, res_col, &term.a, term.a_col, term.a_col, &term.b, term.b_col, term.b_col,
                )
            };
        } else {
            unsafe {
                apply::<BE, E, true, false>(
                    module, cnv_offset, res, res_col, &term.a, term.a_col, term.a_col, &term.b, term.b_col, term.b_col,
                )
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_pairwise_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    i: usize,
    j: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    if i == j {
        unsafe { apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, i, i, b, i, i) };
    } else {
        unsafe { apply::<BE, E, false, true>(module, cnv_offset, res, res_col, a, i, j, b, i, j) };
    }
}

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply<BE, E: TaskExecutor>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        let n = res.n();
        let cols = res.cols();
        let res_ptr = SendPtr(res.raw_mut().as_mut_ptr());
        E::for_each(res_size, |limb| unsafe {
            std::slice::from_raw_parts_mut(res_ptr.get().add(n * (limb * cols + res_col)), n).fill(0)
        });
        return;
    }
    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);
    let n = res.n();
    let cols = res.cols();
    let res_ptr = SendPtr(res.raw_mut().as_mut_ptr());
    E::for_each(res_size, |k| {
        let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (k * cols + res_col)), n) };
        if k < min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            for (coeff, out) in dst.iter_mut().enumerate() {
                *out = (j_min..j_max)
                    .map(|j| a.at(a_col, k_abs - j)[coeff] as i128 * b.at(b_col, j)[b_coeff] as i128)
                    .sum();
            }
        } else {
            dst.fill(0);
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply_add<BE, E: TaskExecutor>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }
    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);
    let n = res.n();
    let cols = res.cols();
    let res_ptr = SendPtr(res.raw_mut().as_mut_ptr());
    E::for_each(min_size, |k| {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (k * cols + res_col)), n) };
        for (coeff, out) in dst.iter_mut().enumerate() {
            *out += (j_min..j_max)
                .map(|j| a.at(a_col, k_abs - j)[coeff] as i128 * b.at(b_col, j)[b_coeff] as i128)
                .sum::<i128>();
        }
    });
}
