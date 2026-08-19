//! Polynomial convolution AVX512 kernels for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! Prepared operands use the packed 2-word group-major layout shared with
//! `VecZnxDft` and `VmpPMat`.

use bytemuck::{cast_slice, cast_slice_mut};
use std::mem::size_of;

use crate::ntt3x42_ifma::{
    kernels::{cond_sub_2q_si512, ntt_avx512},
    module::handle,
    primes::Primes42,
    serial::for_index,
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
};
use poulpy_hal::layouts::{
    CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, DataView, DataViewMut, Module,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
};

use super::{
    mat_vec_ifma::{PrimeConsts512, reduce_bbc_single_prime_512},
    vmp::{pack_y, unpack_y},
};

use crate::NTT3x42Ifma;
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_or_si512, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_stream_si512,
};

// ─────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ─────────────────────────────────────────────────────────────────────────────

/// Output-tile width of the apply kernels (padded window rows on each side).
const TILE: usize = 4;

/// Keep small overwrite results cache-resident for their immediate consumer.
/// Larger outputs use non-temporal stores to avoid evicting the convolution
/// inputs and staging buffers.
#[inline]
fn cached_overwrite_stores(n: usize, size: usize) -> bool {
    const MAX_CACHED_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
    n.checked_mul(size)
        .and_then(|len| len.checked_mul(2 * size_of::<u64>()))
        .is_some_and(|bytes| bytes <= MAX_CACHED_OUTPUT_BYTES)
}

/// Scratch bytes for the packed apply kernels.
pub(crate) fn cnv_apply_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let staged = res_size.min(a_size + b_size).div_ceil(TILE) * TILE;
    (3 * 8 * (a_size + 2 * (TILE - 1)) + 3 * 8 * b_size + 3 * 8 * staged) * size_of::<u64>()
}

/// Scratch bytes for pairwise packed apply.
pub(crate) fn cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        cnv_apply_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }
}

pub(crate) fn cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let staged = res_size.min(a_size + b_size).div_ceil(TILE) * TILE;
    (6 * 8 * (a_size + 2 * (TILE - 1)) + 6 * 8 * b_size + 9 * 8 * staged) * size_of::<u64>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Packed layout helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Zero the packed limb `(col, j)` of a `VecZnxDft`.
#[inline]
fn zero_res_limb(res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>, col: usize, j: usize) {
    let n = res.n();
    let cols = res.cols();
    let off = 2 * n * (j * cols + col);
    let res_u64: &mut [u64] = cast_slice_mut(res.data_mut());
    res_u64[off..off + 2 * n].fill(0);
}

#[inline(always)]
fn col_slice(raw: &[u64], n: usize, size: usize, col: usize) -> &[u64] {
    let stride = 2 * n * size;
    &raw[col * stride..(col + 1) * stride]
}

#[inline(always)]
fn col_slice_mut(raw: &mut [u64], n: usize, size: usize, col: usize) -> &mut [u64] {
    let stride = 2 * n * size;
    &mut raw[col * stride..(col + 1) * stride]
}

/// Offset (in u64) of the packed 2-word row of `(limb_row, group)` in a column.
#[inline(always)]
fn packed_row_offset(size: usize, limb_row: usize, group: usize) -> usize {
    (group * size + limb_row) * 16
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_planar_rank1(
    a0: &[u64],
    a1: &[u64],
    b0: &[u64],
    b1: &[u64],
    win_rows: usize,
    b_size: usize,
    min_size: usize,
    offset: usize,
    staged: usize,
    diag0: &mut [u64],
    pair: &mut [u64],
    diag1: &mut [u64],
) {
    const WIDTH: usize = 2;
    let pad = TILE - 1;
    let n_tiles = min_size.div_ceil(WIDTH);
    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };

    unsafe {
        for (prime, pc_prime) in pc.iter().enumerate() {
            let a0_p = a0.as_ptr().add(prime * 8 * win_rows);
            let a1_p = a1.as_ptr().add(prime * 8 * win_rows);
            let b0_p = b0.as_ptr().add(prime * 8 * b_size);
            let b1_p = b1.as_ptr().add(prime * 8 * b_size);

            for tile in 0..n_tiles {
                let k0 = offset + WIDTH * tile;
                let j_lo = (k0 + 1).saturating_sub(win_rows - 2 * pad).min(b_size);
                let j_hi = (k0 + WIDTH).min(b_size);

                let mut d0_lo0 = _mm512_setzero_si512();
                let mut d0_hi0 = _mm512_setzero_si512();
                let mut d0_lo1 = _mm512_setzero_si512();
                let mut d0_hi1 = _mm512_setzero_si512();
                let mut ps_lo0 = _mm512_setzero_si512();
                let mut ps_hi0 = _mm512_setzero_si512();
                let mut ps_lo1 = _mm512_setzero_si512();
                let mut ps_hi1 = _mm512_setzero_si512();
                let mut d1_lo0 = _mm512_setzero_si512();
                let mut d1_hi0 = _mm512_setzero_si512();
                let mut d1_lo1 = _mm512_setzero_si512();
                let mut d1_hi1 = _mm512_setzero_si512();

                if j_lo < j_hi {
                    let r_start = b_size - j_hi;
                    let r_end = b_size - j_lo;
                    let win_start = 8 * ((k0 + pad + 1) - j_hi);
                    let mut a0_ptr = a0_p.add(win_start);
                    let mut a1_ptr = a1_p.add(win_start);
                    let mut b0_ptr = b0_p.add(8 * r_start);
                    let mut b1_ptr = b1_p.add(8 * r_start);
                    let mut a00 = _mm512_loadu_si512(a0_ptr as *const __m512i);
                    let mut a01 = _mm512_loadu_si512(a0_ptr.add(8) as *const __m512i);
                    let mut a10 = _mm512_loadu_si512(a1_ptr as *const __m512i);
                    let mut a11 = _mm512_loadu_si512(a1_ptr.add(8) as *const __m512i);

                    let mut r = r_start;
                    loop {
                        let y0 = _mm512_loadu_si512(b0_ptr as *const __m512i);
                        let y1 = _mm512_loadu_si512(b1_ptr as *const __m512i);
                        let ys = _mm512_add_epi64(y0, y1);

                        d0_lo0 = _mm512_madd52lo_epu64(d0_lo0, a00, y0);
                        d0_hi0 = _mm512_madd52hi_epu64(d0_hi0, a00, y0);
                        d0_lo1 = _mm512_madd52lo_epu64(d0_lo1, a01, y0);
                        d0_hi1 = _mm512_madd52hi_epu64(d0_hi1, a01, y0);
                        d1_lo0 = _mm512_madd52lo_epu64(d1_lo0, a10, y1);
                        d1_hi0 = _mm512_madd52hi_epu64(d1_hi0, a10, y1);
                        d1_lo1 = _mm512_madd52lo_epu64(d1_lo1, a11, y1);
                        d1_hi1 = _mm512_madd52hi_epu64(d1_hi1, a11, y1);
                        let as0 = _mm512_add_epi64(a00, a10);
                        let as1 = _mm512_add_epi64(a01, a11);
                        ps_lo0 = _mm512_madd52lo_epu64(ps_lo0, as0, ys);
                        ps_hi0 = _mm512_madd52hi_epu64(ps_hi0, as0, ys);
                        ps_lo1 = _mm512_madd52lo_epu64(ps_lo1, as1, ys);
                        ps_hi1 = _mm512_madd52hi_epu64(ps_hi1, as1, ys);

                        r += 1;
                        if r == r_end {
                            break;
                        }
                        a00 = a01;
                        a10 = a11;
                        a0_ptr = a0_ptr.add(8);
                        a1_ptr = a1_ptr.add(8);
                        a01 = _mm512_loadu_si512(a0_ptr.add(8) as *const __m512i);
                        a11 = _mm512_loadu_si512(a1_ptr.add(8) as *const __m512i);
                        b0_ptr = b0_ptr.add(8);
                        b1_ptr = b1_ptr.add(8);
                    }
                }

                let store = |out: &mut [u64], t: usize, lo: __m512i, hi: __m512i| {
                    let value = reduce_bbc_single_prime_512(
                        lo,
                        hi,
                        pc_prime.q,
                        pc_prime.q2,
                        pc_prime.pow42,
                        pc_prime.pow52,
                        pc_prime.pow52_quot,
                    );
                    _mm512_storeu_si512(
                        out.as_mut_ptr().add(prime * 8 * staged + 8 * (WIDTH * tile + t)) as *mut __m512i,
                        value,
                    );
                };
                store(diag0, 0, d0_lo0, d0_hi0);
                store(diag0, 1, d0_lo1, d0_hi1);
                store(pair, 0, ps_lo0, ps_hi0);
                store(pair, 1, ps_lo1, ps_hi1);
                store(diag1, 0, d1_lo0, d1_hi0);
                store(diag1, 1, d1_lo1, d1_hi1);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled column kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve one x8 coefficient group of a column pair.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_columns_packed_group<const ACC: bool, const PAIRWISE: bool>(
    res: &mut [u64],
    res_col: usize,
    n: usize,
    res_cols: usize,
    min_size: usize,
    offset: usize,
    group: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
    cached_overwrite: bool,
    tmp: &mut [u64],
) {
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;
    let n_tiles = min_size.div_ceil(TILE);
    let staged = n_tiles * TILE;
    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let (win, rest) = tmp.split_at_mut(3 * 8 * win_rows);
    let (b_pl, out_st) = rest.split_at_mut(3 * 8 * b_size);
    let out_st = &mut out_st[..3 * 8 * staged];
    let out_base = out_st.as_mut_ptr();

    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
        let zero = _mm512_setzero_si512();
        for p in 0..3 {
            let wp = win.as_mut_ptr().add(p * 8 * win_rows);
            for r in 0..pad {
                _mm512_storeu_si512(wp.add(8 * r) as *mut __m512i, zero);
                _mm512_storeu_si512(wp.add(8 * (a_size + pad + r)) as *mut __m512i, zero);
            }
        }

        let a0_base = a0_col.as_ptr().add(packed_row_offset(a_size, 0, group));
        let a1_base = a1_col.as_ptr().add(packed_row_offset(a_size, 0, group));
        for r in 0..a_size {
            let w0 = _mm512_loadu_si512(a0_base.add(16 * r) as *const __m512i);
            let w1 = _mm512_loadu_si512(a0_base.add(16 * r + 8) as *const __m512i);
            let mut y = unpack_y(w0, w1, m42, m20);
            if PAIRWISE {
                let v0 = _mm512_loadu_si512(a1_base.add(16 * r) as *const __m512i);
                let v1 = _mm512_loadu_si512(a1_base.add(16 * r + 8) as *const __m512i);
                let y1 = unpack_y(v0, v1, m42, m20);
                y = [
                    _mm512_add_epi64(y[0], y1[0]),
                    _mm512_add_epi64(y[1], y1[1]),
                    _mm512_add_epi64(y[2], y1[2]),
                ];
            }
            for (p, yp) in y.iter().enumerate() {
                _mm512_storeu_si512(win.as_mut_ptr().add(p * 8 * win_rows + 8 * (pad + r)) as *mut __m512i, *yp);
            }
        }

        let b0_base = b0_col.as_ptr().add(packed_row_offset(b_size, 0, group));
        let b1_base = b1_col.as_ptr().add(packed_row_offset(b_size, 0, group));
        for r in 0..b_size {
            let w0 = _mm512_loadu_si512(b0_base.add(16 * r) as *const __m512i);
            let w1 = _mm512_loadu_si512(b0_base.add(16 * r + 8) as *const __m512i);
            let mut y = unpack_y(w0, w1, m42, m20);
            if PAIRWISE {
                let v0 = _mm512_loadu_si512(b1_base.add(16 * r) as *const __m512i);
                let v1 = _mm512_loadu_si512(b1_base.add(16 * r + 8) as *const __m512i);
                let y1 = unpack_y(v0, v1, m42, m20);
                y = [
                    _mm512_add_epi64(y[0], y1[0]),
                    _mm512_add_epi64(y[1], y1[1]),
                    _mm512_add_epi64(y[2], y1[2]),
                ];
            }
            for (p, yp) in y.iter().enumerate() {
                _mm512_storeu_si512(b_pl.as_mut_ptr().add(p * 8 * b_size + 8 * r) as *mut __m512i, *yp);
            }
        }

        for (prime, pc_prime) in pc.iter().enumerate() {
            let win_p = win.as_ptr().add(prime * 8 * win_rows);
            let b_base = b_pl.as_ptr().add(prime * 8 * b_size);

            for tile in 0..n_tiles {
                let k0 = offset + TILE * tile;
                let j_lo = (k0 + 1).saturating_sub(a_size).min(b_size);
                let j_hi = (k0 + TILE).min(b_size);

                let mut acc_lo0 = _mm512_setzero_si512();
                let mut acc_hi0 = _mm512_setzero_si512();
                let mut acc_lo1 = _mm512_setzero_si512();
                let mut acc_hi1 = _mm512_setzero_si512();
                let mut acc_lo2 = _mm512_setzero_si512();
                let mut acc_hi2 = _mm512_setzero_si512();
                let mut acc_lo3 = _mm512_setzero_si512();
                let mut acc_hi3 = _mm512_setzero_si512();

                if j_lo < j_hi {
                    // b row r holds limb j = b_size-1-r; iterate r ascending.
                    // w_t = padded window row (k0 + t - j) + pad, sliding up one
                    // row per r and reusing three of four registers.
                    let r_start = b_size - j_hi;
                    let r_end = b_size - j_lo;
                    let mut w_ptr = win_p.add(8 * ((k0 + pad + 1) - j_hi));
                    let mut y_ptr = b_base.add(8 * r_start);

                    let mut w0 = _mm512_loadu_si512(w_ptr as *const __m512i);
                    let mut w1 = _mm512_loadu_si512(w_ptr.add(8) as *const __m512i);
                    let mut w2 = _mm512_loadu_si512(w_ptr.add(16) as *const __m512i);
                    let mut w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);

                    let mut r = r_start;
                    loop {
                        let y = _mm512_loadu_si512(y_ptr as *const __m512i);

                        acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, w0, y);
                        acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, w0, y);
                        acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, w1, y);
                        acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, w1, y);
                        acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, w2, y);
                        acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, w2, y);
                        acc_lo3 = _mm512_madd52lo_epu64(acc_lo3, w3, y);
                        acc_hi3 = _mm512_madd52hi_epu64(acc_hi3, w3, y);

                        r += 1;
                        if r == r_end {
                            break;
                        }
                        w0 = w1;
                        w1 = w2;
                        w2 = w3;
                        w_ptr = w_ptr.add(8);
                        w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);
                        y_ptr = y_ptr.add(8);
                    }
                }

                let store = |t: usize, lo: __m512i, hi: __m512i| {
                    let out = reduce_bbc_single_prime_512(
                        lo,
                        hi,
                        pc_prime.q,
                        pc_prime.q2,
                        pc_prime.pow42,
                        pc_prime.pow52,
                        pc_prime.pow52_quot,
                    );
                    _mm512_storeu_si512(out_base.add(prime * 8 * staged + 8 * (TILE * tile + t)) as *mut __m512i, out);
                };

                store(0, acc_lo0, acc_hi0);
                store(1, acc_lo1, acc_hi1);
                store(2, acc_lo2, acc_hi2);
                store(3, acc_lo3, acc_hi3);
            }
        }

        // Pack the three canonical staging planes into the packed output rows.
        for k_rel in 0..min_size {
            let p0 = _mm512_loadu_si512(out_base.add(8 * k_rel) as *const __m512i);
            let p1 = _mm512_loadu_si512(out_base.add(8 * (staged + k_rel)) as *const __m512i);
            let p2 = _mm512_loadu_si512(out_base.add(8 * (2 * staged + k_rel)) as *const __m512i);
            let dst = res.as_mut_ptr().add((k_rel * res_cols + res_col) * 2 * n + 16 * group);
            if ACC {
                let d = unpack_y(
                    _mm512_loadu_si512(dst as *const __m512i),
                    _mm512_loadu_si512(dst.add(8) as *const __m512i),
                    m42,
                    m20,
                );
                let r = [
                    cond_sub_2q_si512(_mm512_add_epi64(d[0], p0), pc[0].q),
                    cond_sub_2q_si512(_mm512_add_epi64(d[1], p1), pc[1].q),
                    cond_sub_2q_si512(_mm512_add_epi64(d[2], p2), pc[2].q),
                ];
                let [w0, w1] = pack_y(r, m22);
                _mm512_storeu_si512(dst as *mut __m512i, w0);
                _mm512_storeu_si512(dst.add(8) as *mut __m512i, w1);
            } else {
                let [w0, w1] = pack_y([p0, p1, p2], m22);
                if cached_overwrite {
                    _mm512_storeu_si512(dst as *mut __m512i, w0);
                    _mm512_storeu_si512(dst.add(8) as *mut __m512i, w1);
                } else {
                    _mm512_stream_si512(dst as *mut __m512i, w0);
                    _mm512_stream_si512(dst.add(8) as *mut __m512i, w1);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_columns_packed<const ACC: bool, const PAIRWISE: bool>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let win_rows = a_size + 2 * (TILE - 1);
    let n_groups = n / 8;
    let n_tiles = min_size.div_ceil(TILE);
    let task_tmp_len = 3 * 8 * win_rows + 3 * 8 * b_size + 3 * 8 * n_tiles * TILE;
    let res_cols = res.cols();
    let cached_overwrite = cached_overwrite_stores(n, min_size);
    let tmp = &mut tmp[..task_tmp_len];
    let res_data: &mut [u64] = cast_slice_mut(res.data_mut());
    for group in 0..n_groups {
        unsafe {
            conv_columns_packed_group::<ACC, PAIRWISE>(
                res_data,
                res_col,
                n,
                res_cols,
                min_size,
                offset,
                group,
                a0_col,
                a1_col,
                a_size,
                b0_col,
                b1_col,
                b_size,
                cached_overwrite,
                tmp,
            );
        }
    }

    if !ACC {
        for j in min_size..res_size {
            zero_res_limb(res, res_col, j);
        }
        if !cached_overwrite {
            _mm_sfence();
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_tensor_rank1_packed_group(
    res: &mut [u64],
    n: usize,
    res_cols: usize,
    min_size: usize,
    offset: usize,
    group: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
    cached_overwrite: bool,
    tmp: &mut [u64],
) {
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;
    let staged = min_size.div_ceil(TILE) * TILE;
    let (a0_win, rest) = tmp.split_at_mut(3 * 8 * win_rows);
    let (a1_win, rest) = rest.split_at_mut(3 * 8 * win_rows);
    let (b0_pl, rest) = rest.split_at_mut(3 * 8 * b_size);
    let (b1_pl, rest) = rest.split_at_mut(3 * 8 * b_size);
    let (diag0, rest) = rest.split_at_mut(3 * 8 * staged);
    let (pair, diag1) = rest.split_at_mut(3 * 8 * staged);

    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
        let zero = _mm512_setzero_si512();
        for p in 0..3 {
            for win in [&mut *a0_win, &mut *a1_win] {
                let wp = win.as_mut_ptr().add(p * 8 * win_rows);
                for r in 0..pad {
                    _mm512_storeu_si512(wp.add(8 * r) as *mut __m512i, zero);
                    _mm512_storeu_si512(wp.add(8 * (a_size + pad + r)) as *mut __m512i, zero);
                }
            }
        }

        let a0_base = a0_col.as_ptr().add(packed_row_offset(a_size, 0, group));
        let a1_base = a1_col.as_ptr().add(packed_row_offset(a_size, 0, group));
        for r in 0..a_size {
            let a0 = unpack_y(
                _mm512_loadu_si512(a0_base.add(16 * r) as *const __m512i),
                _mm512_loadu_si512(a0_base.add(16 * r + 8) as *const __m512i),
                m42,
                m20,
            );
            let a1 = unpack_y(
                _mm512_loadu_si512(a1_base.add(16 * r) as *const __m512i),
                _mm512_loadu_si512(a1_base.add(16 * r + 8) as *const __m512i),
                m42,
                m20,
            );
            for p in 0..3 {
                _mm512_storeu_si512(
                    a0_win.as_mut_ptr().add(p * 8 * win_rows + 8 * (pad + r)) as *mut __m512i,
                    a0[p],
                );
                _mm512_storeu_si512(
                    a1_win.as_mut_ptr().add(p * 8 * win_rows + 8 * (pad + r)) as *mut __m512i,
                    a1[p],
                );
            }
        }

        let b0_base = b0_col.as_ptr().add(packed_row_offset(b_size, 0, group));
        let b1_base = b1_col.as_ptr().add(packed_row_offset(b_size, 0, group));
        for r in 0..b_size {
            let b0 = unpack_y(
                _mm512_loadu_si512(b0_base.add(16 * r) as *const __m512i),
                _mm512_loadu_si512(b0_base.add(16 * r + 8) as *const __m512i),
                m42,
                m20,
            );
            let b1 = unpack_y(
                _mm512_loadu_si512(b1_base.add(16 * r) as *const __m512i),
                _mm512_loadu_si512(b1_base.add(16 * r + 8) as *const __m512i),
                m42,
                m20,
            );
            for p in 0..3 {
                _mm512_storeu_si512(b0_pl.as_mut_ptr().add(p * 8 * b_size + 8 * r) as *mut __m512i, b0[p]);
                _mm512_storeu_si512(b1_pl.as_mut_ptr().add(p * 8 * b_size + 8 * r) as *mut __m512i, b1[p]);
            }
        }

        conv_planar_rank1(
            a0_win, a1_win, b0_pl, b1_pl, win_rows, b_size, min_size, offset, staged, diag0, pair, diag1,
        );

        for k in 0..min_size {
            let mut d0 = [_mm512_setzero_si512(); 3];
            let mut pairwise = [_mm512_setzero_si512(); 3];
            let mut d1 = [_mm512_setzero_si512(); 3];
            for p in 0..3 {
                let off = p * 8 * staged + 8 * k;
                d0[p] = _mm512_loadu_si512(diag0.as_ptr().add(off) as *const __m512i);
                d1[p] = _mm512_loadu_si512(diag1.as_ptr().add(off) as *const __m512i);
                pairwise[p] = _mm512_loadu_si512(pair.as_ptr().add(off) as *const __m512i);
            }

            for (col, values) in [(0, d0), (1, pairwise), (2, d1)] {
                let [w0, w1] = pack_y(values, m22);
                let dst = res.as_mut_ptr().add((k * res_cols + col) * 2 * n + 16 * group);
                if cached_overwrite {
                    _mm512_storeu_si512(dst as *mut __m512i, w0);
                    _mm512_storeu_si512(dst.add(8) as *mut __m512i, w1);
                } else {
                    _mm512_stream_si512(dst as *mut __m512i, w0);
                    _mm512_stream_si512(dst.add(8) as *mut __m512i, w1);
                }
            }
        }
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_tensor_rank1_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    tmp: &mut [u8],
) {
    assert!(res.cols() >= 3);
    assert!(a.cols() >= 2);
    assert!(b.cols() >= 2);
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for col in 0..3 {
            for j in 0..res_size {
                zero_res_limb(res, col, j);
            }
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let win_rows = a_size + 2 * (TILE - 1);
    let staged = min_size.div_ceil(TILE) * TILE;
    let task_tmp_len = 6 * 8 * win_rows + 6 * 8 * b_size + 9 * 8 * staged;
    let n_groups = n / 8;
    let res_cols = res.cols();
    let cached_overwrite = cached_overwrite_stores(n, min_size);
    let a_raw: &[u64] = cast_slice(a.data());
    let b_raw: &[u64] = cast_slice(b.data());
    let a0 = col_slice(a_raw, n, a_size, 0);
    let a1 = col_slice(a_raw, n, a_size, 1);
    let b0 = col_slice(b_raw, n, b_size, 0);
    let b1 = col_slice(b_raw, n, b_size, 1);
    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let task_tmp = &mut tmp_u64[..task_tmp_len];
    let res_data: &mut [u64] = cast_slice_mut(res.data_mut());
    for group in 0..n_groups {
        unsafe {
            conv_tensor_rank1_packed_group(
                res_data,
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
                cached_overwrite,
                task_tmp,
            );
        }
    }
    if !cached_overwrite {
        _mm_sfence();
    }

    for col in 0..3 {
        for j in min_size..res_size {
            zero_res_limb(res, col, j);
        }
    }
}

/// DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            zero_res_limb(res, res_col, j);
        }
        return;
    }

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a_col_u64 = col_slice(cast_slice(a.data()), n, a_size, a_col);
    let b_col_u64 = col_slice(cast_slice(b.data()), n, b_size, b_col);
    unsafe {
        conv_columns_packed::<false, false>(
            cnv_offset, res, res_col, a_col_u64, a_col_u64, a_size, b_col_u64, b_col_u64, b_size, tmp_u64,
        );
    }
}

/// Accumulating variant of [`cnv_apply_dft_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_dft_accumulate_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a_col_u64 = col_slice(cast_slice(a.data()), n, a_size, a_col);
    let b_col_u64 = col_slice(cast_slice(b.data()), n, b_size, b_col);
    unsafe {
        conv_columns_packed::<true, false>(
            cnv_offset, res, res_col, a_col_u64, a_col_u64, a_size, b_col_u64, b_col_u64, b_size, tmp_u64,
        );
    }
}

/// Pairwise DFT-domain convolution:
/// `res[k] = Σ (a[col_0, k−j] + a[col_1, k−j]) ⊙ (b[col_0, j] + b[col_1, j])`.
///
/// When `col_0 == col_1`, delegates to [`cnv_apply_dft_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_pairwise_apply_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    col_0: usize,
    col_1: usize,
    tmp: &mut [u8],
) {
    if col_0 == col_1 {
        unsafe { cnv_apply_dft_ifma(res, cnv_offset, res_col, a, col_0, b, col_1, tmp) };
        return;
    }

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            zero_res_limb(res, res_col, j);
        }
        return;
    }

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a_u64: &[u64] = cast_slice(a.data());
    let b_u64: &[u64] = cast_slice(b.data());
    let a0 = col_slice(a_u64, n, a_size, col_0);
    let a1 = col_slice(a_u64, n, a_size, col_1);
    let b0 = col_slice(b_u64, n, b_size, col_0);
    let b1 = col_slice(b_u64, n, b_size, col_1);
    unsafe { conv_columns_packed::<false, true>(cnv_offset, res, res_col, a0, a1, a_size, b0, b1, b_size, tmp_u64) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Prepare paths
// ─────────────────────────────────────────────────────────────────────────────

/// Pack one canonical planar NTT-domain limb (`[0, q)` residues) into the
/// packed 2-word rows of `(limb_row, group)` for every group.
#[target_feature(enable = "avx512f")]
unsafe fn pack_limb_packed(dst: &mut [u64], src: &[u64], n: usize, size: usize, limb_row: usize) {
    let n_groups = n / 8;
    // Prepared operands are re-read long after every cache level has turned
    // over; NT stores skip the write-allocate read of the destination lines.
    // Row offsets are multiples of 128 bytes, so base alignment suffices.
    let streamable = dst.as_ptr().addr().is_multiple_of(64);
    let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
    for group in 0..n_groups {
        let src_off = 8 * group;
        unsafe {
            let p0 = _mm512_loadu_si512(src.as_ptr().add(src_off) as *const __m512i);
            let p1 = _mm512_loadu_si512(src.as_ptr().add(n + src_off) as *const __m512i);
            let p2 = _mm512_loadu_si512(src.as_ptr().add(2 * n + src_off) as *const __m512i);
            let w0 = _mm512_or_si512(p0, _mm512_slli_epi64::<42>(_mm512_and_si512(p1, m22)));
            let w1 = _mm512_or_si512(_mm512_srli_epi64::<22>(p1), _mm512_slli_epi64::<20>(p2));
            let dst_off = packed_row_offset(size, limb_row, group);
            if streamable {
                _mm512_stream_si512(dst.as_mut_ptr().add(dst_off) as *mut __m512i, w0);
                _mm512_stream_si512(dst.as_mut_ptr().add(dst_off + 8) as *mut __m512i, w1);
            } else {
                _mm512_storeu_si512(dst.as_mut_ptr().add(dst_off) as *mut __m512i, w0);
                _mm512_storeu_si512(dst.as_mut_ptr().add(dst_off + 8) as *mut __m512i, w1);
            }
        }
    }
    if streamable {
        _mm_sfence();
    }
}

fn zero_limb_packed(dst: &mut [u64], size: usize, limb_row: usize, n_groups: usize) {
    for group in 0..n_groups {
        let off = packed_row_offset(size, limb_row, group);
        dst[off..off + 16].fill(0);
    }
}

/// Scratch bytes required by [`cnv_prepare_left`]: NTT and canonical limbs.
pub(crate) fn cnv_prepare_left_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_left(
    module: &Module<NTT3x42Ifma>,
    res: &mut CnvPVecLBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u8],
) {
    let n = res.n();
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (limb_b, limb_c) = tmp_u64[..6 * n].split_at_mut(3 * n);

    let res_raw: &mut [u64] = cast_slice_mut(res.data_mut());
    for col in 0..cols {
        let dst = col_slice_mut(res_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb_b, a.at(col, j));
            }
            // Lazy [0, 4q): c_from_b re-reduces to the canonical packing domain.
            unsafe { ntt_avx512::<Primes42>(table, limb_b, true) };
            NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(limb_c), limb_b);
            unsafe { pack_limb_packed(dst, limb_c, n, res_size, j) };
        }
        for j in min_size..res_size {
            zero_limb_packed(dst, res_size, j, n / 8);
        }
    }
}

/// Scratch bytes required by [`cnv_prepare_right`]: NTT and canonical limbs.
pub(crate) fn cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_right(
    module: &Module<NTT3x42Ifma>,
    res: &mut CnvPVecRBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u64],
) {
    let n = res.n();
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let (limb_b, limb_c) = tmp[..6 * n].split_at_mut(3 * n);

    let res_raw: &mut [u64] = cast_slice_mut(res.data_mut());
    for col in 0..cols {
        let dst = col_slice_mut(res_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb_b, a.at(col, j));
            }
            // Lazy [0, 4q): c_from_b re-reduces to the canonical packing domain.
            unsafe { ntt_avx512::<Primes42>(table, limb_b, true) };
            NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(limb_c), limb_b);
            unsafe { pack_limb_packed(dst, limb_c, n, res_size, res_size - 1 - j) };
        }
        for j in min_size..res_size {
            zero_limb_packed(dst, res_size, res_size - 1 - j, n / 8);
        }
    }
}

/// Scratch bytes required by [`cnv_prepare_self`]: NTT and canonical limbs.
pub(crate) fn cnv_prepare_self_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_self(
    module: &Module<NTT3x42Ifma>,
    left: &mut CnvPVecLBackendMut<'_, NTT3x42Ifma>,
    right: &mut CnvPVecRBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u8],
) {
    let n = left.n();
    let table = &handle(module).table_ntt;
    let cols = left.cols();
    let res_size = left.size();
    let min_size = res_size.min(a.size());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (limb_b, limb_c) = tmp_u64[..6 * n].split_at_mut(3 * n);

    let left_raw: &mut [u64] = cast_slice_mut(left.data_mut());
    let right_raw: &mut [u64] = cast_slice_mut(right.data_mut());
    for col in 0..cols {
        let dst_l = col_slice_mut(left_raw, n, res_size, col);
        let dst_r = col_slice_mut(right_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb_b, a.at(col, j));
            }
            // Lazy [0, 4q): c_from_b re-reduces to the canonical packing domain.
            unsafe { ntt_avx512::<Primes42>(table, limb_b, true) };
            NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(limb_c), limb_b);
            unsafe { pack_limb_packed(dst_l, limb_c, n, res_size, j) };
            unsafe { pack_limb_packed(dst_r, limb_c, n, res_size, res_size - 1 - j) };
        }
        for j in min_size..res_size {
            zero_limb_packed(dst_l, res_size, j, n / 8);
            zero_limb_packed(dst_r, res_size, res_size - 1 - j, n / 8);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// By-const apply (coefficient domain, unchanged by the prepared layout)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
    b_coeff: usize,
    _tmp: &mut [u8],
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            res.at_mut(res_col, j).fill(0i128);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);
    let n = res.n();
    let rc = res.cols();
    let res_raw = res.raw_mut();

    if b_size == 1 {
        let b0 = b.at(b_col, 0)[b_coeff] as i128;
        for_index(res_size, 2 * n * res_size, |k| {
            let start = n * (k * rc + res_col);
            let res_limb = &mut res_raw[start..start + n];
            let k_abs = k + offset;
            if k < min_size && k_abs < a_size {
                let a_limb = a.at(a_col, k_abs);
                for n_i in 0..res_limb.len() {
                    res_limb[n_i] = (a_limb[n_i] as i128) * b0;
                }
            } else {
                res_limb.fill(0i128);
            }
        });
        return;
    }

    for_index(res_size, 2 * n * res_size * b_size, |k| {
        let start = n * (k * rc + res_col);
        let res_limb = &mut res_raw[start..start + n];
        if k < min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            for (n_i, r) in res_limb.iter_mut().enumerate() {
                let mut acc: i128 = 0;
                for j in j_min..j_max {
                    let b_j = b.at(b_col, j)[b_coeff];
                    acc += a.at(a_col, k_abs - j)[n_i] as i128 * b_j as i128;
                }
                *r = acc;
            }
        } else {
            res_limb.fill(0i128);
        }
    });
}
