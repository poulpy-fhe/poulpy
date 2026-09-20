//! Vector-matrix product AVX512 kernels for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

#![allow(dead_code)]

mod digits4;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, _MM_HINT_T0, _mm_prefetch, _mm_sfence, _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512,
    _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_or_si512, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_stream_si512,
};
use std::arch::x86_64::{_mm256_loadu_si256, _mm512_cvtepu32_epi64, _mm512_permutexvar_epi64};
use std::mem::size_of;

use crate::ntt3x42_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    execution::SendPtr,
    kernels::ntt_avx512,
    module::handle,
    primes::Primes42,
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
};
use poulpy_core::oep::gglwe_product_digit_output_size;
use poulpy_cpu_ref::reference::vmp_select::assert_extractable;
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
        VmpPMatBackendRef, ZnxInfos, check_degree,
    },
};

use super::{
    kernels::{cond_sub_2q_si512, harvey_modmul_si512},
    mat_vec_ifma::PrimeConsts512,
    vec_znx_dft::MASK22,
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
unsafe fn save_planar_digit(dst_base: *mut u64, bq: usize, pc: &[PrimeConsts512; 3], overwrite: bool, red: [__m512i; 3]) {
    unsafe {
        if overwrite {
            save_planar_overwrite(dst_base, bq, red[0], red[1], red[2]);
        } else {
            save_planar_add(dst_base, bq, pc, red[0], red[1], red[2]);
        }
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

/// Row-prime-local VMP prepare, packed.
///
/// Layout: `n_blk_quads × ncols × nrows × 2` groups of 8 u64. Each group pair
/// packs the three 42-bit CRT residues of the row's 8 coefficients into
/// 2 words per coefficient (`w0`, `w1` planes of 8 words each).
///
/// Element `(blk_quad, col, row)` offset in u64:
///   `((blk_quad * ncols + col) * nrows + row) * 16`.
pub(crate) fn vmp_prepare_ifma<E: TaskExecutor>(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VmpPMatBackendMut<'_, crate::NTT3x42Ifma>,
    a: &MatZnxBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    check_degree::<crate::NTT3x42Ifma>(module.n(), n);
    assert_eq!(a.n(), n, "vmp_prepare: a.n():{} != res.n():{n}", a.n());
    let table = handle(module).table_ntt_for(n);
    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blk_quads = n / 8;

    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());
    let pmat_ptr = SendPtr(pmat_u64.as_mut_ptr());

    let bq_stride = ncols * nrows * 16;
    let col_stride = nrows * 16;
    let row_stride = 16;

    E::for_each_chunked(nrows, tmp, 6 * n, |tmp, row_i| {
        let (tmp_b, tmp_c_u64) = tmp.split_at_mut(3 * n);
        let tmp_c_u64 = &mut tmp_c_u64[..3 * n];
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTT3x42Ifma::ntt3x42_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            // Lazy [0, 4q): consumed only by c_from_b (re-reduces).
            unsafe { ntt_avx512::<Primes42>(table, tmp_b, true) };
            let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);
            crate::NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, tmp_c, tmp_b);

            for bq in 0..n_blk_quads {
                let coeff_base = 8 * bq;
                let dst_base = bq * bq_stride + col_i * col_stride + row_i * row_stride;
                let dst = unsafe { std::slice::from_raw_parts_mut(pmat_ptr.get().add(dst_base), 16) };
                for i in 0..8 {
                    let p0 = tmp_c_u64[coeff_base + i];
                    let p1 = tmp_c_u64[n + coeff_base + i];
                    let p2 = tmp_c_u64[2 * n + coeff_base + i];
                    dst[i] = p0 | (p1 & MASK22) << 42;
                    dst[8 + i] = (p1 >> 22) | (p2 << 20);
                }
            }
        }
    });
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
            // Stagger the limb streams to avoid prefetching into the same cache sets.
            let ahead = 2 + (row & 3);
            if bq + ahead < n / 8 {
                _mm_prefetch::<_MM_HINT_T0>(src.add(16 * ahead).cast());
                _mm_prefetch::<_MM_HINT_T0>(src.add(16 * ahead + 8).cast());
            }
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

#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
unsafe fn reduce_vmp(lo: __m512i, hi: __m512i, p: &PrimeConsts512) -> __m512i {
    unsafe {
        let mask = _mm512_set1_epi64((1i64 << 42) - 1);
        let y = _mm512_madd52lo_epu64(_mm512_and_si512(lo, mask), _mm512_srli_epi64::<42>(lo), p.pow42);
        let z = _mm512_madd52lo_epu64(_mm512_and_si512(y, mask), _mm512_srli_epi64::<42>(y), p.pow42);
        let hi = harvey_modmul_si512(hi, p.pow52, p.pow52_quot, p.q);
        // Both terms are below 2q, so the final two subtractions suffice.
        let sum = _mm512_add_epi64(z, hi);
        cond_sub_2q_si512(cond_sub_2q_si512(sum, p.q2), p.q)
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
            reduce_vmp(acc_lo0, acc_hi0, &pc[0]),
            reduce_vmp(acc_lo1, acc_hi1, &pc[1]),
            reduce_vmp(acc_lo2, acc_hi2, &pc[2]),
        ]
    }
}

/// Two right-hand sides against one packed key column.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
unsafe fn madd_reduce_col_x2(
    x0_pm: &[u64],
    x1_pm: &[u64],
    row_max: usize,
    y_base: *const u64,
    pc: &[PrimeConsts512; 3],
) -> [[__m512i; 3]; 2] {
    unsafe {
        let x_plane_sz = 8 * row_max;
        let x00 = x0_pm.as_ptr() as *const __m512i;
        let x01 = x0_pm.as_ptr().add(x_plane_sz) as *const __m512i;
        let x02 = x0_pm.as_ptr().add(2 * x_plane_sz) as *const __m512i;
        let x10 = x1_pm.as_ptr() as *const __m512i;
        let x11 = x1_pm.as_ptr().add(x_plane_sz) as *const __m512i;
        let x12 = x1_pm.as_ptr().add(2 * x_plane_sz) as *const __m512i;
        let m42 = _mm512_set1_epi64(((1u64 << 42) - 1) as i64);
        let m20 = _mm512_set1_epi64(((1u64 << 20) - 1) as i64);

        let mut lo00 = _mm512_setzero_si512();
        let mut hi00 = _mm512_setzero_si512();
        let mut lo01 = _mm512_setzero_si512();
        let mut hi01 = _mm512_setzero_si512();
        let mut lo02 = _mm512_setzero_si512();
        let mut hi02 = _mm512_setzero_si512();
        let mut lo10 = _mm512_setzero_si512();
        let mut hi10 = _mm512_setzero_si512();
        let mut lo11 = _mm512_setzero_si512();
        let mut hi11 = _mm512_setzero_si512();
        let mut lo12 = _mm512_setzero_si512();
        let mut hi12 = _mm512_setzero_si512();

        for r in 0..row_max {
            let y_row = y_base.add(r * 16);
            let w0 = _mm512_loadu_si512(y_row as *const __m512i);
            let w1 = _mm512_loadu_si512(y_row.add(8) as *const __m512i);
            let [y0, y1, y2] = unpack_y(w0, w1, m42, m20);
            let a00 = _mm512_loadu_si512(x00.add(r));
            let a01 = _mm512_loadu_si512(x01.add(r));
            let a02 = _mm512_loadu_si512(x02.add(r));
            let a10 = _mm512_loadu_si512(x10.add(r));
            let a11 = _mm512_loadu_si512(x11.add(r));
            let a12 = _mm512_loadu_si512(x12.add(r));

            lo00 = _mm512_madd52lo_epu64(lo00, a00, y0);
            hi00 = _mm512_madd52hi_epu64(hi00, a00, y0);
            lo01 = _mm512_madd52lo_epu64(lo01, a01, y1);
            hi01 = _mm512_madd52hi_epu64(hi01, a01, y1);
            lo02 = _mm512_madd52lo_epu64(lo02, a02, y2);
            hi02 = _mm512_madd52hi_epu64(hi02, a02, y2);
            lo10 = _mm512_madd52lo_epu64(lo10, a10, y0);
            hi10 = _mm512_madd52hi_epu64(hi10, a10, y0);
            lo11 = _mm512_madd52lo_epu64(lo11, a11, y1);
            hi11 = _mm512_madd52hi_epu64(hi11, a11, y1);
            lo12 = _mm512_madd52lo_epu64(lo12, a12, y2);
            hi12 = _mm512_madd52hi_epu64(hi12, a12, y2);
        }

        [
            [
                reduce_vmp(lo00, hi00, &pc[0]),
                reduce_vmp(lo01, hi01, &pc[1]),
                reduce_vmp(lo02, hi02, &pc[2]),
            ],
            [
                reduce_vmp(lo10, hi10, &pc[0]),
                reduce_vmp(lo11, hi11, &pc[1]),
                reduce_vmp(lo12, hi12, &pc[2]),
            ],
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

                let red0 = reduce_vmp(acc_lo0, acc_hi0, &pc[0]);
                let red1 = reduce_vmp(acc_lo1, acc_hi1, &pc[1]);
                let red2 = reduce_vmp(acc_lo2, acc_hi2, &pc[2]);

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
unsafe fn vmp_apply_core_pm<const OVERWRITE: bool, E: TaskExecutor>(
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

    if !E::is_parallel() || n_blk_quads < 2 {
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
    E::for_each_chunked(n_blk_quads, tmp, 3 * 8 * row_max, |x_pm, bq| {
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
    });

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
pub(crate) fn vmp_apply_dft_to_dft_ifma<E: TaskExecutor>(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    let n = res.n();
    check_degree::<crate::NTT3x42Ifma>(module.n(), n);
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
        vmp_apply_core_pm::<true, E>(
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
pub(crate) fn vmp_apply_dft_to_dft_add_ifma<E: TaskExecutor>(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    let n = res.n();
    check_degree::<crate::NTT3x42Ifma>(module.n(), n);
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
        vmp_apply_core_pm::<false, E>(
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
pub(crate) fn vmp_apply_dft_to_dft_digits_ifma<E: TaskExecutor>(
    _module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    digits: &[VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>],
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let max_size = res.size();
    assert_eq!(max_size, res.size());

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
        let pad = ((dsize - di) as isize - 2).max(0) as usize;
        let res_size_di = res_cols * (max_size - pad);
        let limb_off = di * cols_out;
        a_slices.push(a_u64);
        row_maxs.push(nrows.min(a_size));
        limb_offs.push(limb_off);
        col_maxs.push(ncols.min(res_size_di + limb_off));
    }

    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_cols * max_size];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let res_size_0 = res_cols * (max_size - (dsize as isize - 2).max(0) as usize);
    for col in col_maxs[0]..res_size_0 {
        res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
    }

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let row_max_all = row_maxs.iter().copied().max().unwrap_or(0);
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

    if !E::is_parallel() || n_blk_quads < 2 {
        let (_kernel_output, x_pm) = tmp.split_at_mut(32);
        let x_pm = &mut x_pm[..3 * 8 * row_max_all];
        for bq in 0..n_blk_quads {
            process_bq(x_pm, bq);
        }
    } else {
        E::for_each_chunked(n_blk_quads, tmp, 3 * 8 * row_max_all, process_bq);
    }
}

pub(crate) fn vmp_apply_digits_strided_tmp_bytes_ifma(
    a_cols: usize,
    a_size: usize,
    dsize: usize,
    b_rows: usize,
    b_cols_in: usize,
    workers: usize,
) -> usize {
    let nrows = b_rows * b_cols_in;
    let row_max_all = (0..dsize)
        .map(|di| (a_cols * ((a_size + di) / dsize).min(b_rows)).min(nrows))
        .max()
        .unwrap_or(0);
    let rhs_count = if dsize >= 2 { 2 } else { 1 };
    (4 * dsize + workers * rhs_count * 3 * 8 * row_max_all) * size_of::<u64>()
}

/// Fused multi-digit VMP over strided digit rows.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_ifma<E: TaskExecutor>(
    _module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
) {
    vmp_apply_dft_to_dft_digits_strided_ifma_inner::<E>(res, a, dsize, product_limbs, pmat, None, tmp)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_ifma_known_zero_prefix<E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    zero_prefix: usize,
    tmp: &mut [u64],
) {
    assert!(zero_prefix <= a.size());
    vmp_apply_dft_to_dft_digits_strided_ifma_inner::<E>(res, a, dsize, product_limbs, pmat, Some(zero_prefix), tmp)
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_digits_strided_ifma_inner<E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    zero_prefix: Option<usize>,
    tmp: &mut [u64],
) {
    if E::is_parallel()
        && a.n() >= 8
        && dsize == 4
        && product_limbs >= 3
        && a.cols() == 1
        && res.cols() == 2
        && pmat.cols_in() == 1
        && pmat.cols_out() == 2
        && pmat.rows() <= 8
        && a.size() >= 16
        && res.size() <= 64
    {
        return digits4::apply::<E>(res, a, pmat, zero_prefix, tmp);
    }
    if E::is_parallel() && res.n() >= 65536 && dsize > 1 && a.size() >= 24 && (32..=128).contains(&(res.cols() * res.size())) {
        vmp_apply_dft_to_dft_digits_strided_ifma_impl::<E, true>(res, a, dsize, product_limbs, pmat, zero_prefix, tmp)
    } else {
        vmp_apply_dft_to_dft_digits_strided_ifma_impl::<E, false>(res, a, dsize, product_limbs, pmat, zero_prefix, tmp)
    }
}

pub(crate) struct RotatedOutput<'a> {
    pub(crate) plan: &'a poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan,
    pub(crate) body: VecZnxDftBackendRef<'a, crate::NTT3x42Ifma>,
    pub(crate) output_size: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_rotate_add<E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    tmp: &mut [u64],
    rotated: &RotatedOutput<'_>,
) {
    assert_eq!(res.n(), a.n());
    assert_eq!(res.n(), pmat.n());
    assert_eq!(res.n(), rotated.body.n());
    assert_eq!(rotated.plan.perm.len(), res.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert!(rotated.output_size * res.cols() <= 128);
    assert!(res.size() >= rotated.output_size);
    vmp_digits_loop::<E, true, true>(res, a, dsize, product_limbs, pmat, None, tmp, Some(rotated));
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn emit_rotated(
    dst: *mut u64,
    n: usize,
    cols: usize,
    output_size: usize,
    dest_bq: usize,
    tile: &[u64],
    rotated: &RotatedOutput<'_>,
    pc: &[PrimeConsts512; 3],
) {
    unsafe {
        // An odd Galois multiplier preserves aligned eight-frequency groups in bit-reversed order.
        let perm = &rotated.plan.perm[8 * dest_bq..8 * dest_bq + 8];
        let bq = perm[0] as usize / 8;
        debug_assert!(perm.iter().all(|p| *p as usize / 8 == bq));
        let lanes = _mm512_cvtepu32_epi64(_mm256_loadu_si256(perm.as_ptr().cast()));
        let m42 = _mm512_set1_epi64((1i64 << 42) - 1);
        let m20 = _mm512_set1_epi64((1i64 << 20) - 1);
        let body: &[u64] = cast_slice(rotated.body.data());
        for col in 0..output_size * cols {
            let source = tile.as_ptr().add(16 * col);
            let mut y = unpack_y(
                _mm512_loadu_si512(source.cast()),
                _mm512_loadu_si512(source.add(8).cast()),
                m42,
                m20,
            );
            if col % cols == 0 && col / cols < rotated.body.size() {
                let body = body.as_ptr().add(2 * n * rotated.body.cols() * (col / cols) + 16 * bq);
                let b = unpack_y(
                    _mm512_loadu_si512(body.cast()),
                    _mm512_loadu_si512(body.add(8).cast()),
                    m42,
                    m20,
                );
                for p in 0..3 {
                    y[p] = cond_sub_2q_si512(_mm512_add_epi64(y[p], b[p]), pc[p].q);
                }
            }
            for v in &mut y {
                *v = _mm512_permutexvar_epi64(lanes, *v);
            }
            save_planar_add(dst.add(col * 2 * n), dest_bq, pc, y[0], y[1], y[2]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_digits_strided_ifma_impl<E: TaskExecutor, const TILED: bool>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    zero_prefix: Option<usize>,
    tmp: &mut [u64],
) {
    vmp_digits_loop::<E, TILED, false>(res, a, dsize, product_limbs, pmat, zero_prefix, tmp, None);
}

#[allow(clippy::too_many_arguments)]
fn vmp_digits_loop<E: TaskExecutor, const TILED: bool, const ROTATE: bool>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    zero_prefix: Option<usize>,
    tmp: &mut [u64],
    rotated: Option<&RotatedOutput<'_>>,
) {
    let n = res.n();
    let output_size = if ROTATE { rotated.unwrap().output_size } else { res.size() };

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

    let (digit_meta, tmp) = tmp.split_at_mut(4 * dsize);
    let (row_maxs, digit_meta) = digit_meta.split_at_mut(dsize);
    let (row_starts, digit_meta) = digit_meta.split_at_mut(dsize);
    let (limb_offs, col_maxs) = digit_meta.split_at_mut(dsize);
    for di in 0..dsize {
        let digit_limbs = ((a_size + di) / dsize).min(dnum);
        // Match the reference product: full-width overwrite, then narrowed accumulations.
        let active_size = gglwe_product_digit_output_size(output_size, pmat.size(), dsize, di, product_limbs);
        let limb_off = di * cols_out;
        let row_end = nrows.min(a_cols * digit_limbs);
        let limb_base = dsize - 1 - di;
        let row_start = zero_prefix.map_or_else(
            || {
                (0..row_end)
                    .take_while(|&row| {
                        let flat = (limb_base + (row / a_cols) * dsize) * a_cols + row % a_cols;
                        a_u64[flat * 2 * n..(flat + 1) * 2 * n].iter().all(|&x| x == 0)
                    })
                    .count()
            },
            |prefix| (a_cols * ((prefix + di) / dsize).min(digit_limbs)).min(row_end),
        );
        row_starts[di] = row_start as u64;
        row_maxs[di] = (row_end - row_start) as u64;
        limb_offs[di] = limb_off as u64;
        col_maxs[di] = ncols.min(res_cols * active_size + limb_off) as u64;
    }

    let res_u64: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..2 * n * res_cols * output_size];
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let res_flat = res_cols * output_size;
    if !ROTATE {
        if row_maxs[0] == 0 {
            res_u64.fill(0);
        } else {
            for col in col_maxs[0] as usize..res_flat {
                res_u64[col * 2 * n..(col + 1) * 2 * n].fill(0);
            }
        }
    }
    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let row_max_all = row_maxs.iter().copied().max().unwrap_or(0) as usize;
    let x_words = 3 * 8 * row_max_all;
    let rhs_count = if dsize >= 2 { 2 } else { 1 };
    let task_tmp_len = rhs_count * x_words;
    let res_ptr = SendPtr(res_u64.as_mut_ptr());
    let process_bq = |task_tmp: &mut [u64], dest_bq: usize| {
        let bq = if ROTATE {
            rotated.unwrap().plan.perm[8 * dest_bq] as usize / 8
        } else {
            dest_bq
        };
        let tiled = TILED;
        // Accumulate locally, then stream each output cache line only once.
        let mut output_tile = if tiled { Some([0u64; 16 * 128]) } else { None };
        let dst_ptr = res_ptr;
        let res_ptr = SendPtr(if tiled {
            output_tile.as_mut().unwrap().as_mut_ptr()
        } else {
            dst_ptr.get()
        });
        let output_stride = if tiled { 16 } else { 2 * n };
        let save_bq = if tiled { 0 } else { bq };
        let (x0_pm, x1_pm) = task_tmp[..task_tmp_len].split_at_mut(x_words);
        let mut di = 0;
        while di < dsize {
            let pair = di + 1 < dsize
                && row_maxs[di] != 0
                && row_starts[di] == row_starts[di + 1]
                && row_maxs[di] == row_maxs[di + 1]
                && limb_offs[di + 1] < col_maxs[di].min(col_maxs[di + 1]);

            if pair {
                let row_start = row_starts[di] as usize;
                let row_max = row_maxs[di] as usize;
                let x0_pm = &mut x0_pm[..3 * 8 * row_max];
                let x1_pm = &mut x1_pm[..3 * 8 * row_max];
                unsafe {
                    extract_blk_quad_prime_major_strided(n, row_max, bq, a_u64, a_cols, dsize - 1 - di, dsize, row_start, x0_pm);
                    extract_blk_quad_prime_major_strided(n, row_max, bq, a_u64, a_cols, dsize - 2 - di, dsize, row_start, x1_pm);
                }

                let limb_off0 = limb_offs[di] as usize;
                let limb_off1 = limb_offs[di + 1] as usize;
                let col_max0 = col_maxs[di] as usize;
                let col_max1 = col_maxs[di + 1] as usize;
                let prefix_end = limb_off1.min(col_max0);
                let shared_start = limb_off1;
                let shared_end = col_max0.min(col_max1);
                let tail0_start = prefix_end.max(shared_end);
                let tail1_start = shared_start.max(shared_end);

                for col_pmat in limb_off0..prefix_end {
                    let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;
                    unsafe {
                        let red = madd_reduce_col(x0_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                        let dst = res_ptr.get().add((col_pmat - limb_off0) * output_stride);
                        save_planar_digit(dst, save_bq, &pc, di == 0, red);
                    }
                }

                for col_pmat in shared_start..shared_end {
                    let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;
                    unsafe {
                        let [red0, red1] = madd_reduce_col_x2(x0_pm, x1_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                        let dst0 = res_ptr.get().add((col_pmat - limb_off0) * output_stride);
                        let dst1 = res_ptr.get().add((col_pmat - limb_off1) * output_stride);
                        save_planar_digit(dst0, save_bq, &pc, di == 0, red0);
                        save_planar_add(dst1, save_bq, &pc, red1[0], red1[1], red1[2]);
                    }
                }

                for col_pmat in tail0_start..col_max0 {
                    let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;
                    unsafe {
                        let red = madd_reduce_col(x0_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                        let dst = res_ptr.get().add((col_pmat - limb_off0) * output_stride);
                        save_planar_digit(dst, save_bq, &pc, di == 0, red);
                    }
                }

                for col_pmat in tail1_start..col_max1 {
                    let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;
                    unsafe {
                        let red = madd_reduce_col(x1_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                        let dst = res_ptr.get().add((col_pmat - limb_off1) * output_stride);
                        save_planar_add(dst, save_bq, &pc, red[0], red[1], red[2]);
                    }
                }

                di += 2;
                continue;
            }

            let limb_off = limb_offs[di] as usize;
            let col_max = col_maxs[di] as usize;
            let row_max = row_maxs[di] as usize;
            if limb_off >= col_max || row_max == 0 {
                di += 1;
                continue;
            }
            let row_start = row_starts[di] as usize;
            let x0_pm = &mut x0_pm[..3 * 8 * row_max];
            unsafe {
                extract_blk_quad_prime_major_strided(n, row_max, bq, a_u64, a_cols, dsize - 1 - di, dsize, row_start, x0_pm);
            }

            for col_pmat in limb_off..col_max {
                let y_off = bq * bq_stride + col_pmat * col_stride_y + row_start * 16;
                unsafe {
                    let red = madd_reduce_col(x0_pm, row_max, pmat_u64.as_ptr().add(y_off), &pc);
                    let dst = res_ptr.get().add((col_pmat - limb_off) * output_stride);
                    save_planar_digit(dst, save_bq, &pc, di == 0, red);
                }
            }
            di += 1;
        }
        if ROTATE {
            unsafe {
                emit_rotated(
                    dst_ptr.get(),
                    n,
                    res_cols,
                    output_size,
                    dest_bq,
                    output_tile.as_ref().unwrap(),
                    rotated.unwrap(),
                    &pc,
                );
            }
        } else if tiled {
            for col in 0..res_flat {
                unsafe {
                    let src = output_tile.as_ref().unwrap().as_ptr().add(16 * col);
                    let dst = dst_ptr.get().add(col * 2 * n + 16 * bq);
                    _mm512_stream_si512(dst.cast(), _mm512_loadu_si512(src.cast()));
                    _mm512_stream_si512(dst.add(8).cast(), _mm512_loadu_si512(src.add(8).cast()));
                }
            }
            unsafe {
                _mm_sfence();
            }
        }
    };

    if !E::is_parallel() || n_blk_quads < 2 {
        for bq in 0..n_blk_quads {
            process_bq(tmp, bq);
        }
    } else {
        E::for_each_chunked(n_blk_quads, tmp, task_tmp_len, process_bq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// vmp_zero
// ─────────────────────────────────────────────────────────────────────────────

/// Zero a `VmpPMat<NTT3x42Ifma>`.
pub(crate) fn vmp_zero(res: &mut VmpPMatBackendMut<'_, crate::NTT3x42Ifma>) {
    res.data_mut().as_mut().fill(0);
}

/// Copies rows `first_row + i * row_step` of `a`, truncated to `res.size()`
/// limbs, into rows `i` of `res`, in the packed block-quad prepared layout.
pub(crate) fn vmp_extract_selected_rows_ifma(
    res: &mut VmpPMatBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    first_row: usize,
    row_step: usize,
) {
    assert_extractable(res, a, first_row, row_step);
    let n: usize = a.n();

    let cols_in: usize = a.cols_in();
    let (res_rows, res_nrows, res_ncols) = (res.rows(), res.rows() * cols_in, res.cols_out() * res.size());
    let (a_nrows, a_ncols) = (a.rows() * cols_in, a.cols_out() * a.size());
    let n_blk_quads: usize = n / 8;
    // Same strides as the prepare kernel above, per matrix shape.
    let (res_bq, res_col) = (res_ncols * res_nrows * 16, res_nrows * 16);
    let (a_bq, a_col) = (a_ncols * a_nrows * 16, a_nrows * 16);
    let span: usize = cols_in * 16;

    let src: &[u64] = cast_slice(a.data());
    let dst: &mut [u64] = cast_slice_mut(res.data_mut());
    for bq in 0..n_blk_quads {
        for col in 0..res_ncols {
            let dst_base: usize = bq * res_bq + col * res_col;
            let src_base: usize = bq * a_bq + col * a_col;
            for i in 0..res_rows {
                let (d, s) = (dst_base + i * span, src_base + (first_row + i * row_step) * span);
                dst[d..d + span].copy_from_slice(&src[s..s + span]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::layouts::PrimeSet;

    #[test]
    fn vmp_reduction_matches_wide_remainder() {
        let mut seed = 0x713da891ef42_u64;
        for (prime, q) in Primes42::Q.into_iter().enumerate() {
            let edges = [0, 1, q - 1, q, q + 1, (1 << 52) - 1, u64::MAX - 1, u64::MAX];
            for case in 0..256 {
                let mut lo = [0u64; 8];
                let mut hi = [0u64; 8];
                for lane in 0..8 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    lo[lane] = if case < 8 { edges[lane] } else { seed };
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    hi[lane] = (if case < 8 { edges[case] } else { seed }) & ((1 << 52) - 1);
                }
                let mut actual = [0u64; 8];
                unsafe {
                    let p = PrimeConsts512::new(prime);
                    let r = reduce_vmp(
                        _mm512_loadu_si512(lo.as_ptr().cast()),
                        _mm512_loadu_si512(hi.as_ptr().cast()),
                        &p,
                    );
                    _mm512_storeu_si512(actual.as_mut_ptr().cast(), r);
                }
                for lane in 0..8 {
                    let expected = ((lo[lane] as u128 + ((hi[lane] as u128) << 52)) % q as u128) as u64;
                    assert_eq!(actual[lane], expected, "prime={prime}, case={case}, lane={lane}");
                }
            }
        }
    }
    #[cfg(feature = "enable-rayon")]
    #[test]
    fn streamed_digits_match_serial_with_sparse_inputs() {
        use poulpy_cpu_rayon::RayonTaskExecutor;
        use poulpy_hal::{api::*, execution::SerialTaskExecutor, layouts::*};

        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let n = 64;
            let module = Module::<crate::NTT3x42Ifma>::new(n as u64);
            let mut input = module.vec_znx_dft_alloc(n, 2, 27);
            let mut key = module.vmp_pmat_alloc(n, 7, 2, 2, 33, PrepareHint::Reuse);
            let mut seed = 0x713da891ef42_u64;
            for data in [input.data.as_mut_slice(), key.data_mut().as_mut()] {
                for group in cast_slice_mut::<_, u64>(data).chunks_exact_mut(16) {
                    for lane in 0..8 {
                        let mut y = [0; 3];
                        for (value, q) in y.iter_mut().zip(Primes42::Q) {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            *value = seed % q;
                        }
                        group[lane] = y[0] | (y[1] & ((1 << 22) - 1)) << 42;
                        group[8 + lane] = (y[1] >> 22) | y[2] << 20;
                    }
                }
            }
            for prefix in [0, 1, 26, 27] {
                input.data[..16 * n * 2 * prefix].fill(0);
                for dsize in [2, 3, 4, 5] {
                    for size in [16, 31, 64] {
                        let mut expected = module.vec_znx_dft_alloc(n, 2, size);
                        let mut actual = module.vec_znx_dft_alloc(n, 2, size);
                        let bytes = vmp_apply_digits_strided_tmp_bytes_ifma(2, 27, dsize, 7, 2, 4);
                        let mut scratch = vec![0; bytes / size_of::<u64>()];
                        expected.data.fill(0xa5);
                        actual.data.fill(0xa5);
                        vmp_apply_dft_to_dft_digits_strided_ifma_impl::<SerialTaskExecutor, false>(
                            &mut expected.to_backend_mut(),
                            &input.to_backend_ref(),
                            dsize,
                            3,
                            &key.to_backend_ref(),
                            None,
                            &mut scratch,
                        );
                        vmp_apply_dft_to_dft_digits_strided_ifma_impl::<RayonTaskExecutor, true>(
                            &mut actual.to_backend_mut(),
                            &input.to_backend_ref(),
                            dsize,
                            3,
                            &key.to_backend_ref(),
                            Some(prefix),
                            &mut scratch,
                        );
                        assert_eq!(actual.data, expected.data, "prefix={prefix}, dsize={dsize}, size={size}");
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod rotated_tests {
    use super::*;
    use poulpy_hal::{api::*, execution::SerialTaskExecutor, layouts::*};

    #[test]
    fn rotated_product_matches_product_body_and_automorphism() {
        let n = 64;
        let module = Module::<crate::NTT3x42Ifma>::new(n as u64);
        #[cfg(feature = "enable-rayon")]
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut seed = 0xabc8712934_u64;
        for dsize in [2, 3, 4] {
            for input_size in [5, 16, 27] {
                for key_size in [3, 33] {
                    for output_size in [2, 9, 33] {
                        let output_size = output_size.min(key_size);
                        let mut a = module.vec_znx_dft_alloc(n, 1, input_size);
                        let mut key = module.vmp_pmat_alloc(n, 8, 1, 2, key_size, PrepareHint::Reuse);
                        let mut body = module.vec_znx_dft_alloc(n, 2, 5);
                        let mut initial = module.vec_znx_dft_alloc(n, 2, output_size + 2);
                        for bytes in [
                            a.data.as_mut_slice(),
                            key.data_mut().as_mut(),
                            body.data.as_mut_slice(),
                            initial.data.as_mut_slice(),
                        ] {
                            for group in cast_slice_mut::<_, u64>(bytes).chunks_exact_mut(16) {
                                for lane in 0..8 {
                                    let y: [u64; 3] = std::array::from_fn(|p| {
                                        seed ^= seed << 13;
                                        seed ^= seed >> 7;
                                        seed ^= seed << 17;
                                        if lane == 0 {
                                            Primes42::Q[p] - 1
                                        } else {
                                            seed % Primes42::Q[p]
                                        }
                                    });
                                    group[lane] = y[0] | ((y[1] & ((1 << 22) - 1)) << 42);
                                    group[8 + lane] = (y[1] >> 22) | (y[2] << 20);
                                }
                            }
                        }
                        a.data.as_mut_slice()[..2 * n * size_of::<u64>()].fill(0);
                        let mut product = module.vec_znx_dft_alloc(n, 2, output_size);
                        let mut tmp = vec![0u64; 16 * 1024];
                        vmp_apply_dft_to_dft_digits_strided_ifma_impl::<SerialTaskExecutor, false>(
                            &mut product.to_backend_mut(),
                            &a.to_backend_ref(),
                            dsize,
                            3,
                            &key.to_backend_ref(),
                            None,
                            &mut tmp,
                        );
                        module.vec_znx_dft_add_assign(&mut product.to_backend_mut(), 0, &body.to_backend_ref(), 0);
                        for p in [1, 5, -3] {
                            let plan = module.vec_znx_dft_automorphism_plan(n, p);
                            let mut reference = module.vec_znx_dft_alloc(n, 2, output_size + 2);
                            let mut fused = module.vec_znx_dft_alloc(n, 2, output_size + 2);
                            reference.data.as_mut_slice().copy_from_slice(initial.data.as_slice());
                            fused.data.as_mut_slice().copy_from_slice(initial.data.as_slice());
                            for col in 0..2 {
                                crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_automorphism_add::<SerialTaskExecutor>(
                                    &plan,
                                    &mut reference.to_backend_mut(),
                                    col,
                                    &product.to_backend_ref(),
                                    col,
                                );
                            }
                            let rotated = RotatedOutput {
                                plan: &plan,
                                body: body.to_backend_ref(),
                                output_size,
                            };
                            vmp_rotate_add::<SerialTaskExecutor>(
                                &mut fused.to_backend_mut(),
                                &a.to_backend_ref(),
                                dsize,
                                3,
                                &key.to_backend_ref(),
                                &mut tmp,
                                &rotated,
                            );
                            assert!(
                                reference.data.as_slice() == fused.data.as_slice(),
                                "dsize={dsize} input={input_size} key={key_size} output={output_size} p={p}"
                            );
                            #[cfg(feature = "enable-rayon")]
                            {
                                fused.data.as_mut_slice().copy_from_slice(initial.data.as_slice());
                                pool.install(|| {
                                    vmp_rotate_add::<poulpy_cpu_rayon::RayonTaskExecutor>(
                                        &mut fused.to_backend_mut(),
                                        &a.to_backend_ref(),
                                        dsize,
                                        3,
                                        &key.to_backend_ref(),
                                        &mut tmp,
                                        &rotated,
                                    )
                                });
                                assert!(reference.data.as_slice() == fused.data.as_slice());
                            }
                        }
                    }
                }
            }
        }
    }
}
