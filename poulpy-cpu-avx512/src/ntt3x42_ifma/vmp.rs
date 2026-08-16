//! Vector-matrix product AVX512 kernels for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

#![allow(dead_code)]

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_or_si512, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_stream_si512,
};
use std::mem::size_of;

use crate::ntt3x42_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    kernels::ntt_avx512,
    module::handle,
    primes::Primes42,
    serial::{SendPtr, for_index_with, use_task_split},
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
};
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
    VmpPMatBackendRef, ZnxInfos,
};

use super::{
    kernels::cond_sub_2q_si512,
    mat_vec_ifma::{PrimeConsts512, reduce_bbc_single_prime_512},
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Non-temporal writeback of one x8 output group into a packed `VecZnxDft` limb.
///
/// The reductions are canonical (`[0, q)`), so they are packed directly.
/// `dst_base` points at the packed limb base (`res_u64[col_res * 2 * n]`) and
/// is 64-byte aligned (`VecZnxDft` storage is `DEFAULTALIGN = 64`); group rows
/// sit at 128-byte multiples — safe for `_mm512_stream_si512`. The caller must
/// issue one `_mm_sfence` before any later load from `res`.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn save_planar_overwrite_nt(dst_base: *mut u64, bq: usize, red0: __m512i, red1: __m512i, red2: __m512i) {
    let off = 16 * bq;
    unsafe {
        let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
        let [w0, w1] = pack_y([red0, red1, red2], m22);
        _mm512_stream_si512(dst_base.add(off) as *mut __m512i, w0);
        _mm512_stream_si512(dst_base.add(off + 8) as *mut __m512i, w1);
    }
}

/// Cached read-modify-write accumulate of one x8 output group.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn save_planar_add(dst_base: *mut u64, bq: usize, pc: &[PrimeConsts512; 3], red0: __m512i, red1: __m512i, red2: __m512i) {
    let off = 16 * bq;
    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
        let dst0 = dst_base.add(off) as *mut __m512i;
        let dst1 = dst_base.add(off + 8) as *mut __m512i;
        let d = unpack_y(
            _mm512_loadu_si512(dst0 as *const __m512i),
            _mm512_loadu_si512(dst1 as *const __m512i),
            m42,
            m20,
        );
        let r = [
            cond_sub_2q_si512(_mm512_add_epi64(d[0], red0), pc[0].q),
            cond_sub_2q_si512(_mm512_add_epi64(d[1], red1), pc[1].q),
            cond_sub_2q_si512(_mm512_add_epi64(d[2], red2), pc[2].q),
        ];
        let [w0, w1] = pack_y(r, m22);
        _mm512_storeu_si512(dst0, w0);
        _mm512_storeu_si512(dst1, w1);
    }
}

/// Cached overwrite for fused-digit VMP.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn save_planar_overwrite(dst_base: *mut u64, bq: usize, red0: __m512i, red1: __m512i, red2: __m512i) {
    let off = 16 * bq;
    unsafe {
        let m22 = _mm512_set1_epi64(((1u64 << 22) - 1) as i64);
        let [w0, w1] = pack_y([red0, red1, red2], m22);
        _mm512_storeu_si512(dst_base.add(off) as *mut __m512i, w0);
        _mm512_storeu_si512(dst_base.add(off + 8) as *mut __m512i, w1);
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn save_planar_result<const OVERWRITE: bool>(
    dst_base: *mut u64,
    bq: usize,
    pc: &[PrimeConsts512; 3],
    red0: __m512i,
    red1: __m512i,
    red2: __m512i,
) {
    unsafe {
        if OVERWRITE {
            save_planar_overwrite_nt(dst_base, bq, red0, red1, red2);
        } else {
            save_planar_add(dst_base, bq, pc, red0, red1, red2);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP prepare
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn vmp_prepare_tmp_bytes_ifma(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn vmp_apply_tmp_bytes_ifma(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    // 32 u64 for kernel output (4 blocks × 8 u64)
    // + 3 * 8 * row_max u64 for prime-major x extract (3 primes × 8 u64 × nrows)
    (32 + 3 * 8 * row_max) * size_of::<u64>()
}

/// Packed 42-bit masks: `w0 = p0 | p1_lo22 << 42`, `w1 = p1_hi20 | p2 << 20`.
const MASK22: u64 = (1 << 22) - 1;

/// Row-prime-local VMP prepare, packed.
///
/// Layout: `n_blk_quads × ncols × nrows × 2` groups of 8 u64. Each group pair
/// packs the three 42-bit CRT residues of the row's 8 coefficients into
/// 2 words per coefficient (`w0`, `w1` planes of 8 words each).
///
/// Element `(blk_quad, col, row)` offset in u64:
///   `((blk_quad * ncols + col) * nrows + row) * 16`.
pub(crate) fn vmp_prepare_ifma(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VmpPMatBackendMut<'_, crate::NTT3x42Ifma>,
    a: &MatZnxBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blk_quads = n / 8;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(3 * n);
    let tmp_c_u64 = &mut tmp_c_u64[..3 * n];
    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    let bq_stride = ncols * nrows * 16;
    let col_stride = nrows * 16;
    let row_stride = 16;

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTT3x42Ifma::ntt3x42_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            // Lazy [0, 4q): consumed only by c_from_b (re-reduces).
            unsafe { ntt_avx512::<Primes42>(&handle(module).table_ntt, tmp_b, true) };
            let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);
            crate::NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, tmp_c, tmp_b);

            for bq in 0..n_blk_quads {
                let coeff_base = 8 * bq;
                let dst_base = bq * bq_stride + col_i * col_stride + row_i * row_stride;
                for i in 0..8 {
                    let p0 = tmp_c_u64[coeff_base + i];
                    let p1 = tmp_c_u64[n + coeff_base + i];
                    let p2 = tmp_c_u64[2 * n + coeff_base + i];
                    pmat_u64[dst_base + i] = p0 | (p1 & MASK22) << 42;
                    pmat_u64[dst_base + 8 + i] = (p1 >> 22) | (p2 << 20);
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP apply
// ──────────────────────────────────────────────────────────────────────────────

/// Unpack one packed group into the three 42-bit residues.
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn unpack_y(w0: __m512i, w1: __m512i, m42: __m512i, m20: __m512i) -> [__m512i; 3] {
    [
        _mm512_and_si512(w0, m42),
        _mm512_or_si512(
            _mm512_srli_epi64::<42>(w0),
            _mm512_slli_epi64::<22>(_mm512_and_si512(w1, m20)),
        ),
        _mm512_srli_epi64::<20>(w1),
    ]
}

/// Pack three canonical residue planes into one packed group.
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn pack_y(y: [__m512i; 3], m22: __m512i) -> [__m512i; 2] {
    [
        _mm512_or_si512(y[0], _mm512_slli_epi64::<42>(_mm512_and_si512(y[1], m22))),
        _mm512_or_si512(_mm512_srli_epi64::<22>(y[1]), _mm512_slli_epi64::<20>(y[2])),
    ]
}

/// Extract one packed row into prime-major registers.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_blk_quad_prime_major_row(n: usize, bq: usize, row: usize, a_u64: &[u64]) -> [__m512i; 3] {
    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        let src = a_u64.as_ptr().add(row * 2 * n + 16 * bq);
        let w0 = _mm512_loadu_si512(src as *const __m512i);
        let w1 = _mm512_loadu_si512(src.add(8) as *const __m512i);
        unpack_y(w0, w1, m42, m20)
    }
}

/// Extract packed rows into prime-major planes.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_blk_quad_prime_major(n: usize, row_max: usize, bq: usize, a_u64: &[u64], x_pm: &mut [u64]) {
    let plane_stride = 8 * row_max;

    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        for row in 0..row_max {
            let src = a_u64.as_ptr().add(row * 2 * n + 16 * bq);
            let w0 = _mm512_loadu_si512(src as *const __m512i);
            let w1 = _mm512_loadu_si512(src.add(8) as *const __m512i);
            let y = unpack_y(w0, w1, m42, m20);
            let dst = x_pm.as_mut_ptr().add(row * 8);
            _mm512_storeu_si512(dst as *mut __m512i, y[0]);
            _mm512_storeu_si512(dst.add(plane_stride) as *mut __m512i, y[1]);
            _mm512_storeu_si512(dst.add(2 * plane_stride) as *mut __m512i, y[2]);
        }
    }
}

/// Strided variant of [`extract_blk_quad_prime_major`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_blk_quad_prime_major_strided(
    n: usize,
    row_max: usize,
    bq: usize,
    a_u64: &[u64],
    cols: usize,
    limb_base: usize,
    limb_step: usize,
    row_start: usize,
    x_pm: &mut [u64],
) {
    let plane_stride = 8 * row_max;

    unsafe {
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);
        for row in 0..row_max {
            let logical_row = row_start + row;
            let col = logical_row % cols;
            let digit = logical_row / cols;
            let flat = (limb_base + digit * limb_step) * cols + col;
            let src = a_u64.as_ptr().add(flat * 2 * n + 16 * bq);
            let w0 = _mm512_loadu_si512(src as *const __m512i);
            let w1 = _mm512_loadu_si512(src.add(8) as *const __m512i);
            let y = unpack_y(w0, w1, m42, m20);
            let dst = x_pm.as_mut_ptr().add(row * 8);
            _mm512_storeu_si512(dst as *mut __m512i, y[0]);
            _mm512_storeu_si512(dst.add(plane_stride) as *mut __m512i, y[1]);
            _mm512_storeu_si512(dst.add(2 * plane_stride) as *mut __m512i, y[2]);
        }
    }
}

/// One block-quad/column inner product.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
unsafe fn madd_reduce_col(x_pm: &[u64], row_max: usize, y_base: *const u64, pc: &[PrimeConsts512; 3]) -> [__m512i; 3] {
    unsafe {
        let x_plane_sz = 8 * row_max;

        // Interleave all 3 primes to keep 6 independent MADD52
        // chains in flight (2 per prime across acc_lo/acc_hi),
        // hiding the multiply latency without excess register
        // pressure.
        let x_base0 = x_pm.as_ptr() as *const __m512i;
        let x_base1 = x_pm.as_ptr().add(x_plane_sz) as *const __m512i;
        let x_base2 = x_pm.as_ptr().add(2 * x_plane_sz) as *const __m512i;
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);

        let mut acc_lo0 = _mm512_setzero_si512();
        let mut acc_hi0 = _mm512_setzero_si512();
        let mut acc_lo1 = _mm512_setzero_si512();
        let mut acc_hi1 = _mm512_setzero_si512();
        let mut acc_lo2 = _mm512_setzero_si512();
        let mut acc_hi2 = _mm512_setzero_si512();

        for r in 0..row_max {
            let x0 = _mm512_loadu_si512(x_base0.add(r));
            let y_row = y_base.add(r * 16);
            let w0 = _mm512_loadu_si512(y_row as *const __m512i);
            let x1 = _mm512_loadu_si512(x_base1.add(r));
            let w1 = _mm512_loadu_si512(y_row.add(8) as *const __m512i);
            let x2 = _mm512_loadu_si512(x_base2.add(r));
            let [y0, y1, y2] = unpack_y(w0, w1, m42, m20);
            acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, x0, y0);
            acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, x0, y0);
            acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, x1, y1);
            acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, x1, y1);
            acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, x2, y2);
            acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, x2, y2);
        }

        [
            reduce_bbc_single_prime_512(
                acc_lo0,
                acc_hi0,
                pc[0].q,
                pc[0].q2,
                pc[0].pow42,
                pc[0].pow52,
                pc[0].pow52_quot,
            ),
            reduce_bbc_single_prime_512(
                acc_lo1,
                acc_hi1,
                pc[1].q,
                pc[1].q2,
                pc[1].pow42,
                pc[1].pow52,
                pc[1].pow52_quot,
            ),
            reduce_bbc_single_prime_512(
                acc_lo2,
                acc_hi2,
                pc[2].q,
                pc[2].q2,
                pc[2].pow42,
                pc[2].pow52,
                pc[2].pow52_quot,
            ),
        ]
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
unsafe fn vmp_apply_core_pm_small_rows<const ROWS: usize, const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u64: &[u64],
    limb_offset: usize,
    col_max: usize,
    res_size: usize,
    nrows: usize,
    ncols: usize,
    pc: &[PrimeConsts512; 3],
) {
    unsafe {
        let n_blk_quads = n / 8;
        let bq_stride = ncols * nrows * 16;
        let col_stride_y = nrows * 16;
        let row_stride_y = 16;
        let active_cols = col_max.saturating_sub(limb_offset);
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);

        for bq in 0..n_blk_quads {
            let mut x_rows = [[_mm512_setzero_si512(); 3]; ROWS];
            for (r, x_row) in x_rows.iter_mut().enumerate() {
                *x_row = extract_blk_quad_prime_major_row(n, bq, r, a_u64);
            }

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_base = pmat_u64.as_ptr().add(bq * bq_stride + col_pmat * col_stride_y);

                let mut acc_lo0 = _mm512_setzero_si512();
                let mut acc_hi0 = _mm512_setzero_si512();
                let mut acc_lo1 = _mm512_setzero_si512();
                let mut acc_hi1 = _mm512_setzero_si512();
                let mut acc_lo2 = _mm512_setzero_si512();
                let mut acc_hi2 = _mm512_setzero_si512();

                for (r, x_row) in x_rows.iter().enumerate() {
                    let y_row = y_base.add(r * row_stride_y);
                    let w0 = _mm512_loadu_si512(y_row as *const __m512i);
                    let w1 = _mm512_loadu_si512(y_row.add(8) as *const __m512i);
                    let [y0, y1, y2] = unpack_y(w0, w1, m42, m20);

                    acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, x_row[0], y0);
                    acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, x_row[0], y0);
                    acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, x_row[1], y1);
                    acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, x_row[1], y1);
                    acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, x_row[2], y2);
                    acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, x_row[2], y2);
                }

                let red0 = reduce_bbc_single_prime_512(
                    acc_lo0,
                    acc_hi0,
                    pc[0].q,
                    pc[0].q2,
                    pc[0].pow42,
                    pc[0].pow52,
                    pc[0].pow52_quot,
                );
                let red1 = reduce_bbc_single_prime_512(
                    acc_lo1,
                    acc_hi1,
                    pc[1].q,
                    pc[1].q2,
                    pc[1].pow42,
                    pc[1].pow52,
                    pc[1].pow52_quot,
                );
                let red2 = reduce_bbc_single_prime_512(
                    acc_lo2,
                    acc_hi2,
                    pc[2].q,
                    pc[2].q2,
                    pc[2].pow42,
                    pc[2].pow52,
                    pc[2].pow52_quot,
                );

                let dst_base = res_u64.as_mut_ptr().add(col_res * 2 * n);
                save_planar_result::<OVERWRITE>(dst_base, bq, pc, red0, red1, red2);
            }
        }

        if OVERWRITE {
            for col in active_cols..res_size {
                res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
            }
            _mm_sfence();
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
unsafe fn vmp_apply_core_pm<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u64: &[u64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    _meta: &Bbc126IfmaMeta<Primes42>,
    tmp: &mut [u64],
) {
    if n < 2 {
        return;
    }

    let n_blk_quads = n / 8;
    let a_size = a_u64.len() / (2 * n);
    let res_size = res_u64.len() / (2 * n);
    let row_end = nrows.min(a_size);
    let row_start = a_u64
        .chunks_exact(2 * n)
        .take(row_end)
        .take_while(|row| row.iter().all(|&x| x == 0))
        .count();
    let row_max = row_end - row_start;
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max || row_max == 0 {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };

    // Matrix layout constants
    let bq_stride = ncols * nrows * 16; // u64 per block-quad
    let col_stride_y = nrows * 16; // u64 per column within a block-quad

    let active_cols = col_max.saturating_sub(limb_offset);
    let a_u64 = &a_u64[row_start * 2 * n..];

    if row_start == 0 && row_max == 1 && a_size == 1 && active_cols <= 16 && limb_offset == 0 {
        unsafe {
            vmp_apply_core_pm_small_rows::<1, OVERWRITE>(
                n,
                res_u64,
                a_u64,
                pmat_u64,
                limb_offset,
                col_max,
                res_size,
                nrows,
                ncols,
                &pc,
            );
        }
        return;
    }

    if row_start == 0 && row_max == 2 && a_size == 2 && active_cols <= 16 && limb_offset == 0 {
        unsafe {
            vmp_apply_core_pm_small_rows::<2, OVERWRITE>(
                n,
                res_u64,
                a_u64,
                pmat_u64,
                limb_offset,
                col_max,
                res_size,
                nrows,
                ncols,
                &pc,
            );
        }
        return;
    }

    if row_start == 0 && row_max == 3 && a_size == 3 && active_cols <= 16 && limb_offset == 0 {
        unsafe {
            vmp_apply_core_pm_small_rows::<3, OVERWRITE>(
                n,
                res_u64,
                a_u64,
                pmat_u64,
                limb_offset,
                col_max,
                res_size,
                nrows,
                ncols,
                &pc,
            );
        }
        return;
    }

    if row_start == 0 && row_max == 4 && a_size == 4 && active_cols <= 16 && limb_offset == 0 {
        unsafe {
            vmp_apply_core_pm_small_rows::<4, OVERWRITE>(
                n,
                res_u64,
                a_u64,
                pmat_u64,
                limb_offset,
                col_max,
                res_size,
                nrows,
                ncols,
                &pc,
            );
        }
        return;
    }

    let work = 3 * n * row_max.max(1) * (col_max - limb_offset).max(1);
    if !use_task_split(n_blk_quads, work) {
        // Scratch: 32 u64 reserved for layout compatibility with vmp_apply_tmp_bytes_ifma
        //        + 3 * 8 * row_max u64 for prime-major x extract
        let (_kernel_output, x_pm) = tmp.split_at_mut(32);
        let x_pm = &mut x_pm[..3 * 8 * row_max];

        for bq in 0..n_blk_quads {
            unsafe { extract_blk_quad_prime_major(n, row_max, bq, a_u64, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;

                unsafe {
                    let red = madd_reduce_col(x_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                    let dst_base = res_u64.as_mut_ptr().add(col_res * 2 * n);
                    save_planar_result::<OVERWRITE>(dst_base, bq, &pc, red[0], red[1], red[2]);
                }
            }
        }

        if OVERWRITE {
            let active_cols = col_max.saturating_sub(limb_offset);
            for col in active_cols..res_size {
                res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
            }
            _mm_sfence();
        }
        return;
    }

    let res_ptr = SendPtr(res_u64.as_mut_ptr());
    for_index_with(
        n_blk_quads,
        work,
        || vec![0u64; 3 * 8 * row_max],
        |x_pm, bq| {
            unsafe { extract_blk_quad_prime_major(n, row_max, bq, a_u64, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;

                unsafe {
                    let red = madd_reduce_col(x_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                    let dst_base = res_ptr.get().add(col_res * 2 * n);
                    save_planar_result::<OVERWRITE>(dst_base, bq, &pc, red[0], red[1], red[2]);
                }
            }

            if OVERWRITE {
                _mm_sfence();
            }
        },
    );

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
        }
        _mm_sfence();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public IFMA hooks
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_ifma(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let limb_offset = limb_offset * pmat.cols_out();
    let _ = res_size;

    let res_flat = res.poly_count();
    let a_flat = a.poly_count();
    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_flat];
    let a_u64: &[u64] = &cast_slice::<_, u64>(a.data())[..2 * n * a_flat];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_pm::<true>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset,
            nrows,
            ncols,
            &handle(module).meta_bbc,
            tmp,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_accumulate_ifma(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let limb_offset = limb_offset * pmat.cols_out();
    let _ = res_size;

    let res_flat = res.poly_count();
    let a_flat = a.poly_count();
    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_flat];
    let a_u64: &[u64] = &cast_slice::<_, u64>(a.data())[..2 * n * a_flat];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_pm::<false>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset,
            nrows,
            ncols,
            &handle(module).meta_bbc,
            tmp,
        );
    }
}

/// Fused multi-digit VMP over materialized digit slices.
pub(crate) fn vmp_apply_dft_to_dft_digits_ifma(
    _module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    digits: &[VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>],
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let output_size = res.size();

    let dsize = digits.len();
    if dsize == 0 || n < 2 {
        return;
    }

    let n_blk_quads = n / 8;
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let cols_out = pmat.cols_out();
    let res_cols = res.cols();

    let bq_stride = ncols * nrows * 16;
    let col_stride_y = nrows * 16;

    let mut a_slices: Vec<&[u64]> = Vec::with_capacity(dsize);
    let mut row_maxs: Vec<usize> = Vec::with_capacity(dsize);
    let mut limb_offs: Vec<usize> = Vec::with_capacity(dsize);
    let mut col_maxs: Vec<usize> = Vec::with_capacity(dsize);
    for (di, a) in digits.iter().enumerate() {
        let a_u64: &[u64] = &cast_slice::<_, u64>(a.data())[..2 * n * a.poly_count()];
        let a_size = a_u64.len() / (2 * n);
        // Match the default implementation: the first (overwriting) digit
        // initializes the full destination. Only accumulating digits use the
        // narrowed output view.
        let pad = if di == 0 {
            0
        } else {
            ((dsize - di) as isize - 2).max(0) as usize
        };
        let active_size = if di == 0 {
            output_size
        } else {
            output_size.min(pmat.size().saturating_sub(pad))
        };
        let res_size_di = res_cols * active_size;
        let limb_off = di * cols_out;
        a_slices.push(a_u64);
        row_maxs.push(nrows.min(a_size));
        limb_offs.push(limb_off);
        col_maxs.push(ncols.min(res_size_di + limb_off));
    }

    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_cols * output_size];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let res_flat = res_cols * output_size;
    for col in col_maxs[0]..res_flat {
        res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
    }

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let row_max_all = row_maxs.iter().copied().max().unwrap_or(0);
    let work: usize = (0..dsize)
        .map(|di| 3 * n * row_maxs[di].max(1) * col_maxs[di].saturating_sub(limb_offs[di]).max(1))
        .sum();

    let res_ptr = SendPtr(res_u64.as_mut_ptr());
    let process_bq = |x_pm: &mut [u64], bq: usize| {
        for di in 0..dsize {
            let limb_off = limb_offs[di];
            let col_max = col_maxs[di];
            if limb_off >= col_max {
                continue;
            }
            let row_max = row_maxs[di];
            let x_pm = &mut x_pm[..3 * 8 * row_max];
            unsafe { extract_blk_quad_prime_major(n, row_max, bq, a_slices[di], x_pm) };

            for col_pmat in limb_off..col_max {
                let col_res = col_pmat - limb_off;
                let y_off = bq * bq_stride + col_pmat * col_stride_y;

                unsafe {
                    let red = madd_reduce_col(x_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                    let dst_base = res_ptr.get().add(col_res * 2 * n);
                    if di == 0 {
                        save_planar_overwrite(dst_base, bq, red[0], red[1], red[2]);
                    } else {
                        save_planar_add(dst_base, bq, &pc, red[0], red[1], red[2]);
                    }
                }
            }
        }
    };

    if !use_task_split(n_blk_quads, work) {
        let (_kernel_output, x_pm) = tmp.split_at_mut(32);
        let x_pm = &mut x_pm[..3 * 8 * row_max_all];
        for bq in 0..n_blk_quads {
            process_bq(x_pm, bq);
        }
    } else {
        for_index_with(
            n_blk_quads,
            work,
            || vec![0u64; 3 * 8 * row_max_all],
            |x_pm, bq| process_bq(x_pm, bq),
        );
    }
}

pub(crate) fn vmp_apply_digits_strided_tmp_bytes_ifma(
    a_cols: usize,
    a_size: usize,
    dsize: usize,
    b_rows: usize,
    b_cols_in: usize,
) -> usize {
    let nrows = b_rows * b_cols_in;
    let row_max_all = (0..dsize)
        .map(|di| (a_cols * ((a_size + di) / dsize).min(b_rows)).min(nrows))
        .max()
        .unwrap_or(0);
    (32 + 3 * 8 * row_max_all) * size_of::<u64>()
}

/// Fused multi-digit VMP over strided digit rows.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_ifma(
    _module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let output_size = res.size();

    if dsize == 0 || n < 2 {
        return;
    }

    let n_blk_quads = n / 8;
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let cols_out = pmat.cols_out();
    let res_cols = res.cols();
    let a_cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();

    let bq_stride = ncols * nrows * 16;
    let col_stride_y = nrows * 16;
    let a_u64: &[u64] = &cast_slice::<_, u64>(a.data())[..2 * n * a.poly_count()];

    let mut row_maxs: Vec<usize> = Vec::with_capacity(dsize);
    let mut row_starts: Vec<usize> = Vec::with_capacity(dsize);
    let mut limb_offs: Vec<usize> = Vec::with_capacity(dsize);
    let mut col_maxs: Vec<usize> = Vec::with_capacity(dsize);
    for di in 0..dsize {
        let digit_limbs = ((a_size + di) / dsize).min(dnum);
        // Match the default implementation: the first (overwriting) digit
        // initializes the full destination. Only accumulating digits use the
        // narrowed output view.
        let pad = if di == 0 {
            0
        } else {
            ((dsize - di) as isize - 2).max(0) as usize
        };
        let active_size = if di == 0 {
            output_size
        } else {
            output_size.min(pmat.size().saturating_sub(pad))
        };
        let res_size_di = res_cols * active_size;
        let limb_off = di * cols_out;
        let row_end = nrows.min(a_cols * digit_limbs);
        let limb_base = dsize - 1 - di;
        let row_start = (0..row_end)
            .take_while(|&row| {
                let flat = (limb_base + (row / a_cols) * dsize) * a_cols + row % a_cols;
                a_u64[flat * 2 * n..(flat + 1) * 2 * n].iter().all(|&x| x == 0)
            })
            .count();
        row_starts.push(row_start);
        row_maxs.push(row_end - row_start);
        limb_offs.push(limb_off);
        col_maxs.push(ncols.min(res_size_di + limb_off));
    }

    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_cols * output_size];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let res_flat = res_cols * output_size;
    if row_maxs[0] == 0 {
        res_u64.fill(0);
    } else {
        for col in col_maxs[0]..res_flat {
            res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
        }
    }

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let row_max_all = row_maxs.iter().copied().max().unwrap_or(0);
    let work: usize = (0..dsize)
        .map(|di| 3 * n * row_maxs[di].max(1) * col_maxs[di].saturating_sub(limb_offs[di]).max(1))
        .sum();

    let res_ptr = SendPtr(res_u64.as_mut_ptr());
    let process_bq = |x_pm: &mut [u64], bq: usize| {
        for di in 0..dsize {
            let limb_off = limb_offs[di];
            let col_max = col_maxs[di];
            if limb_off >= col_max {
                continue;
            }
            let row_max = row_maxs[di];
            if row_max == 0 {
                continue;
            }
            let row_start = row_starts[di];
            let x_pm = &mut x_pm[..3 * 8 * row_max];
            unsafe {
                extract_blk_quad_prime_major_strided(n, row_max, bq, a_u64, a_cols, dsize - 1 - di, dsize, row_start, x_pm)
            };

            for col_pmat in limb_off..col_max {
                let col_res = col_pmat - limb_off;
                let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;

                unsafe {
                    let red = madd_reduce_col(x_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                    let dst_base = res_ptr.get().add(col_res * 2 * n);
                    if di == 0 {
                        save_planar_overwrite(dst_base, bq, red[0], red[1], red[2]);
                    } else {
                        save_planar_add(dst_base, bq, &pc, red[0], red[1], red[2]);
                    }
                }
            }
        }
    };

    if !use_task_split(n_blk_quads, work) {
        let (_kernel_output, x_pm) = tmp.split_at_mut(32);
        let x_pm = &mut x_pm[..3 * 8 * row_max_all];
        for bq in 0..n_blk_quads {
            process_bq(x_pm, bq);
        }
    } else {
        for_index_with(
            n_blk_quads,
            work,
            || vec![0u64; 3 * 8 * row_max_all],
            |x_pm, bq| process_bq(x_pm, bq),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// vmp_zero
// ─────────────────────────────────────────────────────────────────────────────

/// Zero a `VmpPMat<NTT3x42Ifma>`.
pub(crate) fn vmp_zero(res: &mut VmpPMatBackendMut<'_, crate::NTT3x42Ifma>) {
    res.data_mut().as_mut().fill(0);
}
