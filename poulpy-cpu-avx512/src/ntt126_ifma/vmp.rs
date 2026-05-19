//! Vector-matrix product AVX512 kernels for [`NTT126Ifma`](crate::NTT126Ifma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi64, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_maskz_permutex2var_epi64, _mm512_permutex2var_epi64, _mm512_set_epi64, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_stream_si512,
};
use std::mem::size_of;

use crate::ntt126_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    module::handle,
    primes::Primes42,
    traits::{Ntt126IfmaCFromB, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64},
    types::Q_SHIFTED_NTT126IFMA,
};
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
    VmpPMatBackendRef, ZnxView, ZnxViewMut,
};

use super::{
    kernels::cond_sub_2q_si512,
    mat_vec_ifma::{PrimeConsts512, reduce_bbc_single_prime_512},
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save helpers
// ──────────────────────────────────────────────────────────────────────────────

fn q2_shifted_vec_512() -> __m512i {
    let q = Q_SHIFTED_NTT126IFMA;
    let q2_512 = [q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]];
    unsafe { _mm512_loadu_si512(q2_512.as_ptr() as *const __m512i) }
}

/// SoA (per-prime) → AoS prep-scalar interleave for one block of a block-quad.
///
/// `red0`/`red1`/`red2` hold the per-prime reductions for the 4 x2-blocks of a
/// block-quad, with lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1, blk2.c0,
/// blk2.c1, blk3.c0, blk3.c1]`. The output for block `I` (0..4) is one prep scalar
/// `__m512i`: `[p0_c0, p1_c0, p2_c0, 0, p0_c1, p1_c1, p2_c1, 0]`.
///
/// Two `vpermi2q` passes materialise the result in a register; keeping the
/// three prime reductions in registers avoids the stack round-trip the previous
/// `save_blk_overwrite` path needed.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn aos_for_blk<const I: usize>(red0: __m512i, red1: __m512i, red2: __m512i) -> __m512i {
    // For block i: keep red0[2i], red1[2i] at lanes 0,1 and red0[2i+1], red1[2i+1] at lanes 4,5.
    // Lanes 2..3, 6..7 are overwritten by the second permute, so indices there are don't-cares.
    let idx01 = match I {
        0 => _mm512_set_epi64(0, 0, 9, 1, 0, 0, 8, 0),
        1 => _mm512_set_epi64(0, 0, 11, 3, 0, 0, 10, 2),
        2 => _mm512_set_epi64(0, 0, 13, 5, 0, 0, 12, 4),
        3 => _mm512_set_epi64(0, 0, 15, 7, 0, 0, 14, 6),
        _ => _mm512_setzero_si512(),
    };
    let tmp01 = _mm512_permutex2var_epi64(red0, idx01, red1);

    // maskz_permutex2var: lanes 0,1,4,5 keep tmp01 (red0/red1 values); lanes 2,6 take red2[2i], red2[2i+1];
    // lanes 3,7 are zeroed by the mask.
    let idxf = match I {
        0 => _mm512_set_epi64(0, 9, 5, 4, 0, 8, 1, 0),
        1 => _mm512_set_epi64(0, 11, 5, 4, 0, 10, 1, 0),
        2 => _mm512_set_epi64(0, 13, 5, 4, 0, 12, 1, 0),
        3 => _mm512_set_epi64(0, 15, 5, 4, 0, 14, 1, 0),
        _ => _mm512_setzero_si512(),
    };
    _mm512_maskz_permutex2var_epi64(0b0111_0111, tmp01, idxf, red2)
}

/// Non-temporal writeback of one SoA→AoS block of a block-quad.
///
/// `dst_base` points at `res_u64[col_res * 4 * n]` and is 64-byte aligned
/// (`VecZnxDft` storage is `DEFAULTALIGN = 64`). Each x2-block stores 8 u64,
/// and `blk` indexes by x2-block, so `dst_base.add(8 * blk)` stays on a
/// 64-byte boundary — safe for `_mm512_stream_si512`. The caller must issue
/// one `_mm_sfence` before any later load from `res`.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn save_blk_overwrite_nt<const I: usize>(dst_base: *mut u64, bq: usize, red0: __m512i, red1: __m512i, red2: __m512i) {
    let out = unsafe { aos_for_blk::<I>(red0, red1, red2) };
    let off = 8 * (bq * 4 + I);
    unsafe {
        _mm512_stream_si512(dst_base.add(off) as *mut __m512i, out);
    }
}

/// Cached load → conditional-subtract-2Q → add → store of one SoA→AoS block.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn save_blk_add<const I: usize>(
    dst_base: *mut u64,
    bq: usize,
    q2_512: __m512i,
    red0: __m512i,
    red1: __m512i,
    red2: __m512i,
) {
    let out = unsafe { aos_for_blk::<I>(red0, red1, red2) };
    let off = 8 * (bq * 4 + I);
    let dst_ptr = unsafe { dst_base.add(off) as *mut __m512i };
    unsafe {
        let d = _mm512_loadu_si512(dst_ptr as *const __m512i);
        let d_red = cond_sub_2q_si512(d, q2_512);
        _mm512_storeu_si512(dst_ptr, _mm512_add_epi64(d_red, out));
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn save_blk_quad_result<const OVERWRITE: bool>(
    dst_base: *mut u64,
    bq: usize,
    q2_512: __m512i,
    red0: __m512i,
    red1: __m512i,
    red2: __m512i,
) {
    unsafe {
        if OVERWRITE {
            save_blk_overwrite_nt::<0>(dst_base, bq, red0, red1, red2);
            save_blk_overwrite_nt::<1>(dst_base, bq, red0, red1, red2);
            save_blk_overwrite_nt::<2>(dst_base, bq, red0, red1, red2);
            save_blk_overwrite_nt::<3>(dst_base, bq, red0, red1, red2);
        } else {
            save_blk_add::<0>(dst_base, bq, q2_512, red0, red1, red2);
            save_blk_add::<1>(dst_base, bq, q2_512, red0, red1, red2);
            save_blk_add::<2>(dst_base, bq, q2_512, red0, red1, red2);
            save_blk_add::<3>(dst_base, bq, q2_512, red0, red1, red2);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP prepare
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn vmp_prepare_tmp_bytes_ifma(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

pub(crate) fn vmp_apply_tmp_bytes_ifma(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    // 32 u64 for kernel output (4 blocks × 8 u64)
    // + 3 * 8 * row_max u64 for prime-major x extract (3 primes × 8 u64 × nrows)
    (32 + 3 * 8 * row_max) * size_of::<u64>()
}

/// Row-prime-local VMP prepare.
///
/// Layout: `n_blk_quads × ncols × nrows × 3` groups of 8 u64. Each group
/// contains one prime's 4 x2-blocks × 2 coefficients for the selected row.
///
/// Element `(blk_quad, col, row, prime)` offset in u64:
///   `((blk_quad * ncols + col) * nrows + row) * 24 + prime * 8`.
pub(crate) fn vmp_prepare_ifma(
    module: &Module<crate::NTT126Ifma>,
    res: &mut VmpPMatBackendMut<'_, crate::NTT126Ifma>,
    a: &MatZnxBackendRef<'_, crate::NTT126Ifma>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blks = n / 2;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);
    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    let bq_stride = ncols * nrows * 24; // u64 per block-quad
    let col_stride = nrows * 24; // u64 per column within a block-quad
    let row_stride = 24; // u64 per row: 3 prime vectors

    // tmp_c is in the interleaved prepared format: per coefficient 4 u64 (as 8 u32),
    // layout [p0, p1, p2, pad] × n coefficients. We scatter each coefficient
    // into row-local prime vectors at the right block-quad slot.
    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTT126Ifma::ntt126_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            crate::NTT126Ifma::ntt126_ifma_dft_execute(&handle(module).table_ntt, tmp_b);
            crate::NTT126Ifma::ntt126_ifma_c_from_b(n, tmp_c, tmp_b);

            let tmp_c_u64: &[u64] = bytemuck::cast_slice(tmp_c);

            for blk_j in 0..n_blks {
                let bq = blk_j / 4;
                let slot = blk_j % 4; // 0..3, maps to lanes 2*slot, 2*slot+1
                let coeff0_base = blk_j * 8; // in tmp_c_u64: 4 u64 per coeff, 2 coeffs per x2-block = 8 u64
                let dst_base = bq * bq_stride + col_i * col_stride + row_i * row_stride + 2 * slot;

                for p in 0..3usize {
                    let dst = dst_base + p * 8;
                    pmat_u64[dst] = tmp_c_u64[coeff0_base + p]; // coeff 0
                    pmat_u64[dst + 1] = tmp_c_u64[coeff0_base + 4 + p]; // coeff 1
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP apply
// ──────────────────────────────────────────────────────────────────────────────

/// SIMD index vectors for `permutex2var` de-interleave of two blocks'
/// `[p0,p1,p2,pad,p0,p1,p2,pad]` data. Used twice per row (blocks 0,1 then 2,3)
/// to fill the low and high halves of each prime-major `__m512i`.
const IDX_PM_P0: [i64; 8] = [0, 4, 8, 12, 0, 0, 0, 0];
const IDX_PM_P1: [i64; 8] = [1, 5, 9, 13, 0, 0, 0, 0];
const IDX_PM_P2: [i64; 8] = [2, 6, 10, 14, 0, 0, 0, 0];

/// Extract a block-quad from interleaved prep scalars into 3 prime-major planes.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_blk_quad_prime_major_row(
    n: usize,
    bq: usize,
    row: usize,
    a_u64: &[u64],
    idx_p0: __m512i,
    idx_p1: __m512i,
    idx_p2: __m512i,
) -> [__m512i; 3] {
    use core::arch::x86_64::{_mm512_castsi512_si256, _mm512_inserti64x4, _mm512_permutex2var_epi64};

    let blk_base = bq * 4 * 2;

    unsafe {
        let src = a_u64.as_ptr().add(row * 4 * n + 4 * blk_base) as *const __m512i;
        let s0 = _mm512_loadu_si512(src);
        let s1 = _mm512_loadu_si512(src.add(1));
        let s2 = _mm512_loadu_si512(src.add(2));
        let s3 = _mm512_loadu_si512(src.add(3));

        let lo_p0 = _mm512_permutex2var_epi64(s0, idx_p0, s1);
        let hi_p0 = _mm512_permutex2var_epi64(s2, idx_p0, s3);
        let p0 = _mm512_inserti64x4::<1>(lo_p0, _mm512_castsi512_si256(hi_p0));

        let lo_p1 = _mm512_permutex2var_epi64(s0, idx_p1, s1);
        let hi_p1 = _mm512_permutex2var_epi64(s2, idx_p1, s3);
        let p1 = _mm512_inserti64x4::<1>(lo_p1, _mm512_castsi512_si256(hi_p1));

        let lo_p2 = _mm512_permutex2var_epi64(s0, idx_p2, s1);
        let hi_p2 = _mm512_permutex2var_epi64(s2, idx_p2, s3);
        let p2 = _mm512_inserti64x4::<1>(lo_p2, _mm512_castsi512_si256(hi_p2));

        [p0, p1, p2]
    }
}

/// Extract a block-quad from interleaved prep scalars into 3 prime-major planes.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn extract_blk_quad_prime_major(n: usize, row_max: usize, bq: usize, a_u64: &[u64], x_pm: &mut [u64]) {
    use core::arch::x86_64::{_mm512_castsi512_si256, _mm512_inserti64x4, _mm512_permutex2var_epi64};

    let plane_stride = 8 * row_max;
    let blk_base = bq * 4 * 2;

    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);

        for row in 0..row_max {
            let src = a_u64.as_ptr().add(row * 4 * n + 4 * blk_base) as *const __m512i;
            let s0 = _mm512_loadu_si512(src);
            let s1 = _mm512_loadu_si512(src.add(1));
            let s2 = _mm512_loadu_si512(src.add(2));
            let s3 = _mm512_loadu_si512(src.add(3));

            let lo_p0 = _mm512_permutex2var_epi64(s0, idx_p0, s1);
            let hi_p0 = _mm512_permutex2var_epi64(s2, idx_p0, s3);
            let p0 = _mm512_inserti64x4::<1>(lo_p0, _mm512_castsi512_si256(hi_p0));

            let lo_p1 = _mm512_permutex2var_epi64(s0, idx_p1, s1);
            let hi_p1 = _mm512_permutex2var_epi64(s2, idx_p1, s3);
            let p1 = _mm512_inserti64x4::<1>(lo_p1, _mm512_castsi512_si256(hi_p1));

            let lo_p2 = _mm512_permutex2var_epi64(s0, idx_p2, s1);
            let hi_p2 = _mm512_permutex2var_epi64(s2, idx_p2, s3);
            let p2 = _mm512_inserti64x4::<1>(lo_p2, _mm512_castsi512_si256(hi_p2));

            let dst = x_pm.as_mut_ptr().add(row * 8);
            _mm512_storeu_si512(dst as *mut __m512i, p0);
            _mm512_storeu_si512(dst.add(plane_stride) as *mut __m512i, p1);
            _mm512_storeu_si512(dst.add(2 * plane_stride) as *mut __m512i, p2);
        }
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
    q2_512: __m512i,
) {
    unsafe {
        let n_blk_quads = n / 8;
        let bq_stride = ncols * nrows * 24;
        let col_stride_y = nrows * 24;
        let row_stride_y = 24;
        let active_cols = col_max.saturating_sub(limb_offset);
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);

        for bq in 0..n_blk_quads {
            let mut x_rows = [[_mm512_setzero_si512(); 3]; ROWS];
            for (r, x_row) in x_rows.iter_mut().enumerate() {
                *x_row = extract_blk_quad_prime_major_row(n, bq, r, a_u64, idx_p0, idx_p1, idx_p2);
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
                    let y0 = _mm512_loadu_si512(y_row as *const __m512i);
                    let y1 = _mm512_loadu_si512(y_row.add(8) as *const __m512i);
                    let y2 = _mm512_loadu_si512(y_row.add(16) as *const __m512i);

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

                let dst_base = res_u64.as_mut_ptr().add(col_res * 4 * n);
                save_blk_quad_result::<OVERWRITE>(dst_base, bq, q2_512, red0, red1, red2);
            }
        }

        if OVERWRITE {
            for col in active_cols..res_size {
                res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
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

    let n_blks = n / 2;
    let n_blk_quads = n_blks / 4;
    let a_size = a_u64.len() / (4 * n);
    let res_size = res_u64.len() / (4 * n);
    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let q2_512 = if OVERWRITE {
        _mm512_setzero_si512()
    } else {
        q2_shifted_vec_512()
    };

    // Matrix layout constants
    let bq_stride = ncols * nrows * 24; // u64 per block-quad
    let col_stride_y = nrows * 24; // u64 per column within a block-quad
    let row_stride_y = 24; // u64 per row: 3 prime vectors

    let active_cols = col_max.saturating_sub(limb_offset);

    if row_max == 1 && a_size == 1 && active_cols <= 16 && limb_offset == 0 {
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
                q2_512,
            );
        }
        return;
    }

    if row_max == 2 && a_size == 2 && active_cols <= 16 && limb_offset == 0 {
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
                q2_512,
            );
        }
        return;
    }

    if row_max == 3 && a_size == 3 && active_cols <= 16 && limb_offset == 0 {
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
                q2_512,
            );
        }
        return;
    }

    if row_max == 4 && a_size == 4 && active_cols <= 16 && limb_offset == 0 {
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
                q2_512,
            );
        }
        return;
    }

    // Scratch: 32 u64 reserved for layout compatibility with vmp_apply_tmp_bytes_ifma
    //        + 3 * 8 * row_max u64 for prime-major x extract
    let (_kernel_output, x_pm) = tmp.split_at_mut(32);
    let x_pm = &mut x_pm[..3 * 8 * row_max];

    for bq in 0..n_blk_quads {
        unsafe { extract_blk_quad_prime_major(n, row_max, bq, a_u64, x_pm) };

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bq * bq_stride + col_pmat * col_stride_y;
            let x_plane_sz = 8 * row_max;

            unsafe {
                let mut red = [_mm512_setzero_si512(); 3];

                // Interleave all 3 primes so Zen 5's two FMA ports always
                // have 6 independent MADD52 chains in flight (2 per prime
                // across acc_lo/acc_hi). This hides the 5-cycle VPMADD52
                // latency without blowing up register pressure.
                let x_base0 = x_pm.as_ptr() as *const __m512i;
                let x_base1 = x_pm.as_ptr().add(x_plane_sz) as *const __m512i;
                let x_base2 = x_pm.as_ptr().add(2 * x_plane_sz) as *const __m512i;
                let y_base = pmat_u64.as_ptr().add(y_off);

                let mut acc_lo0 = _mm512_setzero_si512();
                let mut acc_hi0 = _mm512_setzero_si512();
                let mut acc_lo1 = _mm512_setzero_si512();
                let mut acc_hi1 = _mm512_setzero_si512();
                let mut acc_lo2 = _mm512_setzero_si512();
                let mut acc_hi2 = _mm512_setzero_si512();

                for r in 0..row_max {
                    let x0 = _mm512_loadu_si512(x_base0.add(r));
                    let y_row = y_base.add(r * row_stride_y);
                    let y0 = _mm512_loadu_si512(y_row as *const __m512i);
                    let x1 = _mm512_loadu_si512(x_base1.add(r));
                    let y1 = _mm512_loadu_si512(y_row.add(8) as *const __m512i);
                    let x2 = _mm512_loadu_si512(x_base2.add(r));
                    let y2 = _mm512_loadu_si512(y_row.add(16) as *const __m512i);
                    acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, x0, y0);
                    acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, x0, y0);
                    acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, x1, y1);
                    acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, x1, y1);
                    acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, x2, y2);
                    acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, x2, y2);
                }

                red[0] = reduce_bbc_single_prime_512(
                    acc_lo0,
                    acc_hi0,
                    pc[0].q,
                    pc[0].q2,
                    pc[0].pow42,
                    pc[0].pow52,
                    pc[0].pow52_quot,
                );
                red[1] = reduce_bbc_single_prime_512(
                    acc_lo1,
                    acc_hi1,
                    pc[1].q,
                    pc[1].q2,
                    pc[1].pow42,
                    pc[1].pow52,
                    pc[1].pow52_quot,
                );
                red[2] = reduce_bbc_single_prime_512(
                    acc_lo2,
                    acc_hi2,
                    pc[2].q,
                    pc[2].q2,
                    pc[2].pow42,
                    pc[2].pow52,
                    pc[2].pow52_quot,
                );

                // SoA → AoS: interleave 3 prime results into 4 prep-scalar blocks
                // directly in SIMD registers (no stack round-trip).
                let base = col_res * 4 * n;
                let dst_base = res_u64.as_mut_ptr().add(base);
                if OVERWRITE {
                    save_blk_overwrite_nt::<0>(dst_base, bq, red[0], red[1], red[2]);
                    save_blk_overwrite_nt::<1>(dst_base, bq, red[0], red[1], red[2]);
                    save_blk_overwrite_nt::<2>(dst_base, bq, red[0], red[1], red[2]);
                    save_blk_overwrite_nt::<3>(dst_base, bq, red[0], red[1], red[2]);
                } else {
                    save_blk_add::<0>(dst_base, bq, q2_512, red[0], red[1], red[2]);
                    save_blk_add::<1>(dst_base, bq, q2_512, red[0], red[1], red[2]);
                    save_blk_add::<2>(dst_base, bq, q2_512, red[0], red[1], red[2]);
                    save_blk_add::<3>(dst_base, bq, q2_512, red[0], red[1], red[2]);
                }
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        _mm_sfence();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public IFMA hooks
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_ifma(
    module: &Module<crate::NTT126Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT126Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT126Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT126Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let limb_offset = limb_offset * pmat.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
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
    module: &Module<crate::NTT126Ifma>,
    res: &mut VecZnxDftBackendMut<'_, crate::NTT126Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT126Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT126Ifma>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let nrows = pmat.rows() * pmat.cols_in();
    let ncols = pmat.cols_out() * pmat.size();
    let limb_offset = limb_offset * pmat.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
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

// ─────────────────────────────────────────────────────────────────────────────
// vmp_zero
// ─────────────────────────────────────────────────────────────────────────────

/// Zero a `VmpPMat<NTT126Ifma>`.
pub(crate) fn vmp_zero(res: &mut VmpPMatBackendMut<'_, crate::NTT126Ifma>) {
    res.data_mut().as_mut().fill(0);
}
