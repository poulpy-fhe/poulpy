use crate::NTT4x30Avx512Backend;
use poulpy_cpu_ref::ring::CpuRing;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_cvtsi64_si128, _mm_loadu_si128, _mm256_broadcastsi128_si256, _mm256_loadu_si256,
    _mm256_storeu_si256, _mm512_add_epi64, _mm512_and_si512, _mm512_cvtepi64_epi32, _mm512_cvtepu32_epi64, _mm512_loadu_si512,
    _mm512_mul_epu32, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_srl_epi64, _mm512_srli_epi64,
};
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_cpu_ref::reference::sparse_log_gap;
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::{
    CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, DataView, DataViewMut, Module,
    VecZnxBackendRef, VecZnxDftBackendMut, ZnxView, check_degree,
};
use std::mem::size_of;

use super::{
    arithmetic_avx512::{BARRETT_MU, POW32, Q_VEC, bcast_quad, cond_sub_512, reduce_b_to_canonical_512},
    vec_znx_dft::packed_limb_mut,
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

/// u32 offset, inside a prepared right operand column whose own degree is
/// `N >> log_gap`, of row `row0` of the slot that degree-`N` slot
/// `group * GROUP + slot_in_group` reads: the slot itself when dense, slot
/// `>> log_gap` when sparse (spec 4.5). Consecutive rows of that slot are
/// `4 * GROUP` u32 apart.
#[inline(always)]
fn slot_offset(size: usize, row0: usize, group: usize, slot_in_group: usize, log_gap: usize) -> usize {
    let slot = (group * GROUP + slot_in_group) >> log_gap;
    packed_row_offset(size, row0, slot / GROUP) + 4 * (slot % GROUP)
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

fn zero_res_limb<R: CpuRing>(res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>, col: usize, limb: usize) {
    let (n, cols) = (res.n(), res.cols());
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    packed_limb_mut(data, n, cols, col, limb).fill(0);
}

#[inline(always)]
unsafe fn reduce_accum(meta: &BbcMeta<Primes30>, lo: __m512i, hi: __m512i) -> __m512i {
    unsafe {
        let mask = _mm512_set1_epi64(((1u64 << meta.h) - 1) as i64);
        let s2l = bcast_quad(meta.s2l_pow_red.as_ptr());
        let s2h = bcast_quad(meta.s2h_pow_red.as_ptr());
        let hi_lo = _mm512_and_si512(hi, mask);
        let hi_hi = _mm512_srl_epi64(hi, _mm_cvtsi64_si128(meta.h as i64));
        let x = _mm512_add_epi64(lo, _mm512_mul_epu32(hi_lo, s2l));
        let x = _mm512_add_epi64(x, _mm512_mul_epu32(hi_hi, s2h));
        reduce_b_to_canonical_512(
            x,
            bcast_quad(Q_VEC.as_ptr()),
            bcast_quad(BARRETT_MU.as_ptr()),
            bcast_quad(POW32.as_ptr()),
        )
    }
}

#[inline(always)]
unsafe fn accumulate_product(lo: &mut __m512i, hi: &mut __m512i, a: __m512i, b: __m512i) {
    unsafe {
        let mask32 = _mm512_set1_epi64(u32::MAX as i64);
        let product = _mm512_mul_epu32(a, b);
        *lo = _mm512_add_epi64(*lo, _mm512_and_si512(product, mask32));
        *hi = _mm512_add_epi64(*hi, _mm512_srli_epi64::<32>(product));
    }
}

#[inline(always)]
unsafe fn load_pair(row: *const u32, pair: usize) -> __m512i {
    unsafe { _mm512_cvtepu32_epi64(_mm256_loadu_si256(row.add(8 * pair) as *const __m256i)) }
}

/// The four residues of one slot duplicated into both halves of a pair: the two
/// slots of a pair of a sparse right operand read the same degree-`n` slot (their
/// degree-`N` indices differ in the low bit, which `>> log_gap` drops).
#[inline(always)]
unsafe fn load_slot_dup(slot: *const u32) -> __m512i {
    unsafe { _mm512_cvtepu32_epi64(_mm256_broadcastsi128_si256(_mm_loadu_si128(slot as *const __m128i))) }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn conv_group<const ACC: bool, const PAIRWISE: bool, const B_SPARSE: bool>(
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
    b_log_gap: usize,
) {
    unsafe {
        let q = bcast_quad(Q_VEC.as_ptr());
        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            for pair in 0..GROUP / 2 {
                let b_off = if B_SPARSE {
                    slot_offset(b_size, b_start, group, 2 * pair, b_log_gap)
                } else {
                    packed_row_offset(b_size, b_start, group) + 8 * pair
                };
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for row in 0..j_max - j_min {
                    let ao = packed_row_offset(a_size, a_start + row, group);
                    let ro = row * 4 * GROUP;
                    let mut av = load_pair(a0.as_ptr().add(ao), pair);
                    let mut bv = if B_SPARSE {
                        load_slot_dup(b0.as_ptr().add(b_off + ro))
                    } else {
                        load_pair(b0.as_ptr().add(b_off + ro), 0)
                    };
                    if PAIRWISE {
                        let bv1 = if B_SPARSE {
                            load_slot_dup(b1.as_ptr().add(b_off + ro))
                        } else {
                            load_pair(b1.as_ptr().add(b_off + ro), 0)
                        };
                        av = cond_sub_512(_mm512_add_epi64(av, load_pair(a1.as_ptr().add(ao), pair)), q);
                        bv = cond_sub_512(_mm512_add_epi64(bv, bv1), q);
                    }
                    accumulate_product(&mut lo, &mut hi, av, bv);
                }
                let value = reduce_accum(meta, lo, hi);
                let dst = res.get().add((k * res_cols + res_col) * 4 * n + group * 4 * GROUP + pair * 8);
                if ACC {
                    let old = _mm512_cvtepu32_epi64(_mm256_loadu_si256(dst as *const __m256i));
                    _mm256_storeu_si256(
                        dst as *mut __m256i,
                        _mm512_cvtepi64_epi32(cond_sub_512(_mm512_add_epi64(old, value), q)),
                    );
                } else {
                    _mm256_storeu_si256(dst as *mut __m256i, _mm512_cvtepi64_epi32(value));
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn conv_rank1_group(
    meta: &BbcMeta<Primes30>,
    res: SendPtr<u32>,
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
        let q = bcast_quad(Q_VEC.as_ptr());
        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            for pair in 0..GROUP / 2 {
                let mut d0_lo = _mm512_setzero_si512();
                let mut d0_hi = _mm512_setzero_si512();
                let mut ps_lo = _mm512_setzero_si512();
                let mut ps_hi = _mm512_setzero_si512();
                let mut d1_lo = _mm512_setzero_si512();
                let mut d1_hi = _mm512_setzero_si512();
                for row in 0..j_max - j_min {
                    let ao = packed_row_offset(a_size, a_start + row, group);
                    let bo = packed_row_offset(b_size, b_start + row, group);
                    let av0 = load_pair(a0.as_ptr().add(ao), pair);
                    let av1 = load_pair(a1.as_ptr().add(ao), pair);
                    let bv0 = load_pair(b0.as_ptr().add(bo), pair);
                    let bv1 = load_pair(b1.as_ptr().add(bo), pair);
                    accumulate_product(&mut d0_lo, &mut d0_hi, av0, bv0);
                    accumulate_product(&mut d1_lo, &mut d1_hi, av1, bv1);
                    accumulate_product(
                        &mut ps_lo,
                        &mut ps_hi,
                        cond_sub_512(_mm512_add_epi64(av0, av1), q),
                        cond_sub_512(_mm512_add_epi64(bv0, bv1), q),
                    );
                }
                for (col, value) in [
                    (0, reduce_accum(meta, d0_lo, d0_hi)),
                    (1, reduce_accum(meta, ps_lo, ps_hi)),
                    (2, reduce_accum(meta, d1_lo, d1_hi)),
                ] {
                    let dst = res.get().add((k * res_cols + col) * 4 * n + group * 4 * GROUP + pair * 8);
                    _mm256_storeu_si256(dst as *mut __m256i, _mm512_cvtepi64_epi32(value));
                }
            }
        }
    }
}

pub(crate) fn cnv_prepare_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

#[target_feature(enable = "avx512f")]
unsafe fn pack_prepared_limb(dst: &mut [u32], src: &[u64], n: usize, size: usize, limb: usize) {
    unsafe {
        let q = bcast_quad(Q_VEC.as_ptr());
        let mu = bcast_quad(BARRETT_MU.as_ptr());
        let pow32 = bcast_quad(POW32.as_ptr());
        for group in 0..n / GROUP {
            let dst_off = packed_row_offset(size, limb, group);
            for pair in 0..GROUP / 2 {
                let src_off = group * 4 * GROUP + pair * 8;
                let x = _mm512_loadu_si512(src.as_ptr().add(src_off) as *const __m512i);
                let x = reduce_b_to_canonical_512(x, q, mu, pow32);
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(dst_off + pair * 8) as *mut __m256i,
                    _mm512_cvtepi64_epi32(x),
                );
            }
        }
    }
}

fn prepare<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    left: Option<&mut CnvPVecLBackendMut<'_, NTT4x30Avx512Backend<R>>>,
    right: Option<&mut CnvPVecRBackendMut<'_, NTT4x30Avx512Backend<R>>>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512Backend<R>>,
    tmp: &mut [u64],
) {
    poulpy_hal::layouts::assert_dense(a, "prepare");
    let (n, cols, size) = if let Some(res) = left.as_ref() {
        (res.n(), res.cols(), res.size())
    } else {
        let res = right.as_ref().unwrap();
        (res.n(), res.cols(), res.size())
    };
    check_degree::<NTT4x30Avx512Backend<R>>(module.n(), n);
    assert_eq!(a.n(), n, "prepare: a.n():{} != res.n():{n}", a.n());
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());
    if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
        assert_eq!(r.n(), n, "prepare: right.n():{} != left.n():{n}", r.n());
        assert_eq!(r.cols(), l.cols(), "right.cols():{} != left.cols():{}", r.cols(), l.cols());
        assert_eq!(r.size(), l.size(), "right.size():{} != left.size():{}", r.size(), l.size());
    }
    let table = module.get_ntt_table_for(n);
    let min_size = size.min(a.size());
    let mut left = left.map(|res| cast_slice_mut::<_, u32>(res.data_mut()));
    let mut right = right.map(|res| cast_slice_mut::<_, u32>(res.data_mut()));
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
                NTT4x30Avx512Backend::<R>::ntt_from_znx64(tmp, a.at(col, limb));
                NTT4x30Avx512Backend::<R>::ntt_dft_execute(table, tmp);
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
            NTT4x30Avx512Backend::<R>::ntt_from_znx64(tmp, a.at(col, limb));
            NTT4x30Avx512Backend::<R>::ntt_dft_execute(table, tmp);
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

pub(crate) fn cnv_prepare_left<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    res: &mut CnvPVecLBackendMut<'_, NTT4x30Avx512Backend<R>>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512Backend<R>>,
    tmp: &mut [u64],
) {
    prepare::<E, _>(module, Some(res), None, a, tmp);
}

pub(crate) fn cnv_prepare_right<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    res: &mut CnvPVecRBackendMut<'_, NTT4x30Avx512Backend<R>>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512Backend<R>>,
    tmp: &mut [u64],
) {
    prepare::<E, _>(module, None, Some(res), a, tmp);
}

pub(crate) fn cnv_prepare_self<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    left: &mut CnvPVecLBackendMut<'_, NTT4x30Avx512Backend<R>>,
    right: &mut CnvPVecRBackendMut<'_, NTT4x30Avx512Backend<R>>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512Backend<R>>,
    tmp: &mut [u64],
) {
    prepare::<E, _>(module, Some(left), Some(right), a, tmp);
}

pub(crate) fn cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
unsafe fn apply<E: TaskExecutor, const ACC: bool, const PAIRWISE: bool, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    a0_col: usize,
    a1_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b0_col: usize,
    b1_col: usize,
) {
    let (n, res_size, a_size, b_size) = (res.n(), res.size(), a.size(), b.size());
    check_degree::<NTT4x30Avx512Backend<R>>(module.n(), n);
    assert_eq!(a.n(), n, "a.n():{} != res.n():{n}", a.n());
    let b_log_gap = sparse_log_gap(n, b.n());
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
    let a_raw: &[u32] = cast_slice(a.data());
    let b_raw: &[u32] = cast_slice(b.data());
    let a0 = col_slice(a_raw, n, a_size, a0_col);
    let a1 = col_slice(a_raw, n, a_size, a1_col);
    let b0 = col_slice(b_raw, b.n(), b_size, b0_col);
    let b1 = col_slice(b_raw, b.n(), b_size, b1_col);
    let res_cols = res.cols();
    let res_ptr = SendPtr(cast_slice_mut::<_, u32>(res.data_mut()).as_mut_ptr());
    E::for_each(n / GROUP, |group| unsafe {
        let meta = module.get_bbc_meta();
        if b_log_gap > 0 {
            conv_group::<ACC, PAIRWISE, true>(
                meta, res_ptr, res_col, n, res_cols, min_size, offset, group, a0, a1, a_size, b0, b1, b_size, b_log_gap,
            )
        } else {
            conv_group::<ACC, PAIRWISE, false>(
                meta, res_ptr, res_col, n, res_cols, min_size, offset, group, a0, a1, a_size, b0, b1, b_size, b_log_gap,
            )
        }
    });
    if !ACC {
        for limb in min_size..res_size {
            zero_res_limb(res, res_col, limb);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b_col: usize,
) {
    unsafe { apply::<E, false, false, _>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft_add<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b_col: usize,
) {
    unsafe { apply::<E, true, false, _>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_pairwise_apply_dft<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
    i: usize,
    j: usize,
) {
    if i == j {
        unsafe { apply::<E, false, false, _>(module, cnv_offset, res, res_col, a, i, i, b, i, i) };
    } else {
        unsafe { apply::<E, false, true, _>(module, cnv_offset, res, res_col, a, i, j, b, i, j) };
    }
}

pub(crate) unsafe fn cnv_tensor_rank1_dft<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
) {
    assert!(res.cols() >= 3 && a.cols() >= 2 && b.cols() >= 2);
    assert_eq!(a.n(), res.n(), "cnv_tensor_rank1_dft: a.n():{} != res.n():{}", a.n(), res.n());
    assert_eq!(b.n(), res.n(), "cnv_tensor_rank1_dft: b.n():{} != res.n():{}", b.n(), res.n());
    let (n, res_size, a_size, b_size) = (res.n(), res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for col in 0..3 {
            for limb in 0..res_size {
                zero_res_limb(res, col, limb);
            }
        }
        return;
    }
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let a_raw: &[u32] = cast_slice(a.data());
    let b_raw: &[u32] = cast_slice(b.data());
    let (a0, a1) = (col_slice(a_raw, n, a_size, 0), col_slice(a_raw, n, a_size, 1));
    let (b0, b1) = (col_slice(b_raw, n, b_size, 0), col_slice(b_raw, n, b_size, 1));
    let res_cols = res.cols();
    let res_ptr = SendPtr(cast_slice_mut::<_, u32>(res.data_mut()).as_mut_ptr());
    E::for_each(n / GROUP, |group| unsafe {
        conv_rank1_group(
            module.get_bbc_meta(),
            res_ptr,
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
    for col in 0..3 {
        for limb in min_size..res_size {
            zero_res_limb(res, col, limb);
        }
    }
}

pub(crate) fn cnv_tensor_rank1_dft_avx512_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

pub(crate) unsafe fn cnv_tensor_rank1_dft_avx512<E: TaskExecutor, R: CpuRing>(
    module: &Module<NTT4x30Avx512Backend<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512Backend<R>>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512Backend<R>>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512Backend<R>>,
    _tmp: &mut [u8],
) {
    unsafe { cnv_tensor_rank1_dft::<E, _>(module, res, cnv_offset, a, b) };
}

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}
