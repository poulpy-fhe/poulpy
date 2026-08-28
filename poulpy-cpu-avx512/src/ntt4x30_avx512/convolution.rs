use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm_cvtsi64_si128, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi64, _mm512_and_si512,
    _mm512_cvtepi64_epi32, _mm512_cvtepu32_epi64, _mm512_loadu_si512, _mm512_mul_epu32, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_srl_epi64, _mm512_srli_epi64,
};
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{
    CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, DataView, DataViewMut, Module,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
};
use std::mem::size_of;

use super::{
    NTT4x30Avx512,
    arithmetic_avx512::{BARRETT_MU, POW32, Q_VEC, bcast_quad, cond_sub_512, reduce_b_to_canonical_512},
    vec_znx_dft::packed_limb_mut,
};

const GROUP: usize = 8;

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

fn zero_res_limb(res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>, col: usize, limb: usize) {
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

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn conv_group<const ACC: bool, const PAIRWISE: bool>(
    meta: &BbcMeta<Primes30>,
    res: &mut [u32],
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
        let q = bcast_quad(Q_VEC.as_ptr());
        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            for pair in 0..GROUP / 2 {
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for row in 0..j_max - j_min {
                    let ao = packed_row_offset(a_size, a_start + row, group);
                    let bo = packed_row_offset(b_size, b_start + row, group);
                    let mut av = load_pair(a0.as_ptr().add(ao), pair);
                    let mut bv = load_pair(b0.as_ptr().add(bo), pair);
                    if PAIRWISE {
                        av = cond_sub_512(_mm512_add_epi64(av, load_pair(a1.as_ptr().add(ao), pair)), q);
                        bv = cond_sub_512(_mm512_add_epi64(bv, load_pair(b1.as_ptr().add(bo), pair)), q);
                    }
                    accumulate_product(&mut lo, &mut hi, av, bv);
                }
                let value = reduce_accum(meta, lo, hi);
                let dst = res
                    .as_mut_ptr()
                    .add((k * res_cols + res_col) * 4 * n + group * 4 * GROUP + pair * 8);
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
    res: &mut [u32],
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
                    let dst = res
                        .as_mut_ptr()
                        .add((k * res_cols + col) * 4 * n + group * 4 * GROUP + pair * 8);
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

fn prepare(
    module: &Module<NTT4x30Avx512>,
    left: Option<&mut CnvPVecLBackendMut<'_, NTT4x30Avx512>>,
    right: Option<&mut CnvPVecRBackendMut<'_, NTT4x30Avx512>>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    mask: i64,
    tmp: &mut [u64],
) {
    let (n, cols, size) = if let Some(res) = left.as_ref() {
        (res.n(), res.cols(), res.size())
    } else {
        let res = right.as_ref().unwrap();
        (res.n(), res.cols(), res.size())
    };
    let min_size = size.min(a.size());
    let mut left = left.map(|res| cast_slice_mut::<_, u32>(res.data_mut()));
    let mut right = right.map(|res| cast_slice_mut::<_, u32>(res.data_mut()));
    for col in 0..cols {
        let mut dst_l = left.as_deref_mut().map(|data| col_slice_mut(data, n, size, col));
        let mut dst_r = right.as_deref_mut().map(|data| col_slice_mut(data, n, size, col));
        for limb in 0..min_size {
            if limb + 1 == min_size {
                NTT4x30Avx512::ntt_from_znx64_masked(tmp, a.at(col, limb), mask);
            } else {
                NTT4x30Avx512::ntt_from_znx64(tmp, a.at(col, limb));
            }
            NTT4x30Avx512::ntt_dft_execute(module.get_ntt_table(), tmp);
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

pub(crate) fn cnv_prepare_left(
    module: &Module<NTT4x30Avx512>,
    res: &mut CnvPVecLBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    mask: i64,
    tmp: &mut [u64],
) {
    prepare(module, Some(res), None, a, mask, tmp);
}

pub(crate) fn cnv_prepare_right(
    module: &Module<NTT4x30Avx512>,
    res: &mut CnvPVecRBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    mask: i64,
    tmp: &mut [u64],
) {
    prepare(module, None, Some(res), a, mask, tmp);
}

pub(crate) fn cnv_prepare_self(
    module: &Module<NTT4x30Avx512>,
    left: &mut CnvPVecLBackendMut<'_, NTT4x30Avx512>,
    right: &mut CnvPVecRBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    mask: i64,
    tmp: &mut [u64],
) {
    prepare(module, Some(left), Some(right), a, mask, tmp);
}

pub(crate) fn cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
unsafe fn apply<const ACC: bool, const PAIRWISE: bool>(
    module: &Module<NTT4x30Avx512>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    a0_col: usize,
    a1_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    b0_col: usize,
    b1_col: usize,
) {
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
    let a_raw: &[u32] = cast_slice(a.data());
    let b_raw: &[u32] = cast_slice(b.data());
    let a0 = col_slice(a_raw, n, a_size, a0_col);
    let a1 = col_slice(a_raw, n, a_size, a1_col);
    let b0 = col_slice(b_raw, n, b_size, b0_col);
    let b1 = col_slice(b_raw, n, b_size, b1_col);
    let res_cols = res.cols();
    let res_raw: &mut [u32] = cast_slice_mut(res.data_mut());
    for group in 0..n / GROUP {
        unsafe {
            conv_group::<ACC, PAIRWISE>(
                module.get_bbc_meta(),
                res_raw,
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
        };
    }
    if !ACC {
        for limb in min_size..res_size {
            zero_res_limb(res, res_col, limb);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft(
    module: &Module<NTT4x30Avx512>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    b_col: usize,
) {
    unsafe { apply::<false, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft_accumulate(
    module: &Module<NTT4x30Avx512>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    b_col: usize,
) {
    unsafe { apply::<true, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_pairwise_apply_dft(
    module: &Module<NTT4x30Avx512>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    i: usize,
    j: usize,
) {
    if i == j {
        unsafe { apply::<false, false>(module, cnv_offset, res, res_col, a, i, i, b, i, i) };
    } else {
        unsafe { apply::<false, true>(module, cnv_offset, res, res_col, a, i, j, b, i, j) };
    }
}

pub(crate) unsafe fn cnv_tensor_rank1_dft(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
) {
    assert!(res.cols() >= 3 && a.cols() >= 2 && b.cols() >= 2);
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
    let res_raw: &mut [u32] = cast_slice_mut(res.data_mut());
    for group in 0..n / GROUP {
        unsafe {
            conv_rank1_group(
                module.get_bbc_meta(),
                res_raw,
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
        };
    }
    for col in 0..3 {
        for limb in min_size..res_size {
            zero_res_limb(res, col, limb);
        }
    }
}

pub(crate) fn cnv_tensor_rank1_dft_avx512_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

pub(crate) unsafe fn cnv_tensor_rank1_dft_avx512(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    _tmp: &mut [u8],
) {
    unsafe { cnv_tensor_rank1_dft(module, res, cnv_offset, a, b) };
}

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    b_col: usize,
    b_coeff: usize,
) {
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for limb in 0..res_size {
            res.at_mut(res_col, limb).fill(0);
        }
        return;
    }
    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);
    for k in 0..res_size {
        if k < min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            for (coeff, out) in res.at_mut(res_col, k).iter_mut().enumerate() {
                *out = (j_min..j_max)
                    .map(|j| a.at(a_col, k_abs - j)[coeff] as i128 * b.at(b_col, j)[b_coeff] as i128)
                    .sum();
            }
        } else {
            res.at_mut(res_col, k).fill(0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply_add(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Avx512>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Avx512>,
    b_col: usize,
    b_coeff: usize,
) {
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }
    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);
    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        for (coeff, out) in res.at_mut(res_col, k).iter_mut().enumerate() {
            *out += (j_min..j_max)
                .map(|j| a.at(a_col, k_abs - j)[coeff] as i128 * b.at(b_col, j)[b_coeff] as i128)
                .sum::<i128>();
        }
    }
}
