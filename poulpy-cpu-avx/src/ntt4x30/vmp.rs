//! Vector-matrix product AVX2 kernels for [`NTT4x30Avx`](crate::NTT4x30Avx).
//!
//! Uses a backend-local prime-major prepared-matrix layout so the hot AVX VMP
//! path streams one prime plane at a time and reuses extracted input rows
//! across the output-column loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_set_epi64x, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use poulpy_cpu_ref::reference::ntt4x30::{
    NttCFromB, NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, types::Q_SHIFTED, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{
    DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    ZnxView, ZnxViewMut,
};

use super::mat_vec_avx::vec_mat1col_product_blkpair_bbc_pm_avx2;
use crate::NTT4x30Avx;

/// Scratch space (in bytes) required by the AVX VMP prepare kernel.
pub(crate) fn vmp_prepare_tmp_bytes_avx(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// AVX-local VMP prepare into a 4-plane prime-major layout.
///
/// The prepared matrix uses one plane per CRT prime. Within each plane the
/// layout is `block_pair -> output_column -> input_row`, and every row stores
/// four u64 values in lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
pub(crate) fn vmp_prepare_avx_pm(
    module: &Module<NTT4x30Avx>,
    res: &mut VmpPMatBackendMut<'_, NTT4x30Avx>,
    a: &MatZnxBackendRef<'_, NTT4x30Avx>,
    tmp: &mut [u64],
) {
    let n = res.n();

    debug_assert_eq!(a.n(), n);
    debug_assert_eq!(res.cols_in(), a.cols_in());
    debug_assert_eq!(res.rows(), a.rows());
    debug_assert_eq!(res.cols_out(), a.cols_out());
    debug_assert_eq!(res.size(), a.size());
    debug_assert!(std::mem::size_of_val(tmp) >= vmp_prepare_tmp_bytes_avx(n));
    debug_assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_block_pairs = n / 4;
    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);

    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            NTT4x30Avx::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT4x30Avx::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            NTT4x30Avx::ntt_c_from_b(n, tmp_c, tmp_b);
            let tmp_c_u64: &[u64] = cast_slice(tmp_c);

            for bp in 0..n_block_pairs {
                let coeff_base = 16 * bp;
                for p in 0..4usize {
                    let dst = p * plane_stride + bp * bp_stride + col_i * col_stride + row_i * 4;
                    pmat_u64[dst..dst + 4].copy_from_slice(&[
                        tmp_c_u64[coeff_base + p],
                        tmp_c_u64[coeff_base + 4 + p],
                        tmp_c_u64[coeff_base + 8 + p],
                        tmp_c_u64[coeff_base + 12 + p],
                    ]);
                }
            }
        }
    }
}

/// Scratch space (in bytes) required by the AVX VMP apply kernels.
pub(crate) fn vmp_apply_tmp_bytes_avx(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    (16 + 16 * row_max) * size_of::<u64>()
}

/// Extract one q120b x2-block from a contiguous q120b matrix.
///
/// Copies one 64-byte block per row using two AVX2 loads and stores.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn extract_1blk_from_contiguous_q120b_avx2(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(n >= 2);
    debug_assert!(n.is_power_of_two());
    debug_assert!(blk < n / 2);
    debug_assert!(src.len() >= row_max * 4 * n);
    debug_assert!(dst.len() >= row_max * 8);

    let src_row_stride = 4 * n;
    let src_blk_off = 8 * blk;

    for row in 0..row_max {
        let src_ptr = unsafe { src.as_ptr().add(row * src_row_stride + src_blk_off) as *const __m256i };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(8 * row) as *mut __m256i };
        unsafe {
            _mm256_storeu_si256(dst_ptr, _mm256_loadu_si256(src_ptr));
            _mm256_storeu_si256(dst_ptr.add(1), _mm256_loadu_si256(src_ptr.add(1)));
        }
    }
}

/// Extract one q120b block pair into 4 prime-major planes.
///
/// Each plane stores `row_max` rows of 4 u64 with lane order
/// `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
#[target_feature(enable = "avx2")]
unsafe fn extract_blk_pair_prime_major_avx2(n: usize, row_max: usize, blk_pair: usize, src: &[u64], dst: &mut [u64]) {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(src.len() >= row_max * 4 * n);
    debug_assert!(dst.len() >= 16 * row_max);

    let plane_stride = 4 * row_max;
    let coeff_base = 16 * blk_pair;

    for row in 0..row_max {
        let row_base = row * 4 * n + coeff_base;
        for p in 0..4usize {
            let dst_ptr = unsafe { dst.as_mut_ptr().add(p * plane_stride + row * 4) as *mut __m256i };
            let plane = _mm256_set_epi64x(
                src[row_base + 12 + p] as i64,
                src[row_base + 8 + p] as i64,
                src[row_base + 4 + p] as i64,
                src[row_base + p] as i64,
            );
            unsafe { _mm256_storeu_si256(dst_ptr, plane) };
        }
    }
}

/// Strided-digit variant of [`extract_blk_pair_prime_major_avx2`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn extract_blk_pair_prime_major_strided_avx2(
    n: usize,
    row_max: usize,
    blk_pair: usize,
    src: &[u64],
    cols: usize,
    limb_base: usize,
    limb_step: usize,
    row_start: usize,
    dst: &mut [u64],
) {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(dst.len() >= 16 * row_max);

    let plane_stride = 4 * row_max;
    let coeff_base = 16 * blk_pair;
    for row in 0..row_max {
        let logical_row = row_start + row;
        let col = logical_row % cols;
        let digit = logical_row / cols;
        let flat = (limb_base + digit * limb_step) * cols + col;
        let row_base = flat * 4 * n + coeff_base;
        for p in 0..4usize {
            let dst_ptr = unsafe { dst.as_mut_ptr().add(p * plane_stride + row * 4) as *mut __m256i };
            let plane = _mm256_set_epi64x(
                src[row_base + 12 + p] as i64,
                src[row_base + 8 + p] as i64,
                src[row_base + 4 + p] as i64,
                src[row_base + p] as i64,
            );
            unsafe { _mm256_storeu_si256(dst_ptr, plane) };
        }
    }
}

/// Overwrite one x2-block (8 u64) of a q120b vector.
#[target_feature(enable = "avx2")]
unsafe fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m256i };
    let src_ptr = src.as_ptr() as *const __m256i;
    unsafe {
        _mm256_storeu_si256(dst_ptr, _mm256_loadu_si256(src_ptr));
        _mm256_storeu_si256(dst_ptr.add(1), _mm256_loadu_si256(src_ptr.add(1)));
    }
}

// Inputs MUST be in `[0, 2q)`, so one unsigned conditional subtract reduces them.
#[target_feature(enable = "avx2")]
unsafe fn save_blk_add(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    unsafe {
        let q = _mm256_loadu_si256(Q_SHIFTED.as_ptr() as *const __m256i);
        let one = _mm256_set1_epi64x(1);
        let q_minus_one = _mm256_sub_epi64(q, one);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let q_minus_one_signed = _mm256_xor_si256(q_minus_one, sign);
        let base = dst.as_mut_ptr().add(8 * blk);

        for half in 0..2 {
            let dp = base.add(4 * half) as *mut __m256i;
            let sp = src.as_ptr().add(4 * half) as *const __m256i;
            let dv = _mm256_loadu_si256(dp as *const __m256i);
            let sv = _mm256_loadu_si256(sp);
            let dm = _mm256_cmpgt_epi64(_mm256_xor_si256(dv, sign), q_minus_one_signed);
            let sm = _mm256_cmpgt_epi64(_mm256_xor_si256(sv, sign), q_minus_one_signed);
            let dr = _mm256_sub_epi64(dv, _mm256_and_si256(dm, q));
            let sr = _mm256_sub_epi64(sv, _mm256_and_si256(sm, q));
            _mm256_storeu_si256(dp, _mm256_add_epi64(dr, sr));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn vmp_apply_core_avx_pm<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u64: &[u64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcMeta<Primes30>,
    tmp: &mut [u64],
) {
    debug_assert!(n >= 4);
    debug_assert!(n.is_power_of_two());
    debug_assert!(n.is_multiple_of(4));

    let a_size = a_u64.len() / (4 * n);
    let res_size = res_u64.len() / (4 * n);
    let n_block_pairs = n / 4;

    let row_end = nrows.min(a_size);
    let row_start = a_u64
        .chunks_exact(4 * n)
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

    let (blkpair_output, x_pm) = tmp.split_at_mut(16);
    let x_pm = &mut x_pm[..16 * row_max];
    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;
    let a_u64 = &a_u64[row_start * 4 * n..];

    for bp in 0..n_block_pairs {
        unsafe { extract_blk_pair_prime_major_avx2(n, row_max, bp, a_u64, x_pm) };

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;

            unsafe {
                vec_mat1col_product_blkpair_bbc_pm_avx2(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], plane_stride)
            };

            let blk0 = 2 * bp;
            let blk1 = blk0 + 1;
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]) };
                unsafe { save_blk_overwrite(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]) };
            } else {
                unsafe { save_blk_add(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]) };
                unsafe { save_blk_add(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]) };
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

pub(crate) fn vmp_apply_dft_to_dft_avx(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.raw());

    unsafe {
        vmp_apply_core_avx_pm::<true>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}

pub(crate) fn vmp_apply_dft_to_dft_accumulate_avx(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.raw());

    unsafe {
        vmp_apply_core_avx_pm::<false>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}

pub(crate) fn vmp_apply_digits_strided_tmp_bytes_avx(
    a_cols: usize,
    a_size: usize,
    dsize: usize,
    b_rows: usize,
    b_cols_in: usize,
) -> usize {
    let nrows = b_rows * b_cols_in;
    let row_max = (0..dsize)
        .map(|di| (a_cols * ((a_size + di) / dsize).min(b_rows)).min(nrows))
        .max()
        .unwrap_or(0);
    (4 * dsize + 16 + 16 * row_max) * size_of::<u64>()
}

/// Applies all gadget digits directly from their interleaved source limbs.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_avx(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    dsize: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let output_size = res.size();
    if dsize == 0 || n < 4 {
        return;
    }

    let n_block_pairs = n / 4;
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let cols_out = pmat.cols_out();
    let res_cols = res.cols();
    let a_cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();
    let meta = module.get_bbc_meta();
    let a_u64: &[u64] = cast_slice(a.raw());
    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let pmat_u64: &[u64] = cast_slice(pmat.raw());

    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    let (digit_meta, tmp) = tmp.split_at_mut(4 * dsize);
    let (row_maxs, digit_meta) = digit_meta.split_at_mut(dsize);
    let (row_starts, digit_meta) = digit_meta.split_at_mut(dsize);
    let (limb_offsets, col_maxs) = digit_meta.split_at_mut(dsize);
    for di in 0..dsize {
        let digit_limbs = ((a_size + di) / dsize).min(dnum);
        // Match the reference product: full-width overwrite, then narrowed accumulations.
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
        let limb_offset = di * cols_out;
        let row_end = (a_cols * digit_limbs).min(nrows);
        let limb_base = dsize - 1 - di;
        let row_start = (0..row_end)
            .take_while(|&row| {
                let flat = (limb_base + (row / a_cols) * dsize) * a_cols + row % a_cols;
                a_u64[flat * 4 * n..(flat + 1) * 4 * n].iter().all(|&x| x == 0)
            })
            .count();
        row_starts[di] = row_start as u64;
        row_maxs[di] = (row_end - row_start) as u64;
        limb_offsets[di] = limb_offset as u64;
        col_maxs[di] = ncols.min(res_cols * active_size + limb_offset) as u64;
    }

    let res_flat = res_cols * output_size;
    if row_maxs[0] == 0 {
        res_u64.fill(0);
    } else {
        for col in col_maxs[0] as usize..res_flat {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }

    let (blkpair_output, x_pm) = tmp.split_at_mut(16);
    for bp in 0..n_block_pairs {
        for di in 0..dsize {
            let limb_offset = limb_offsets[di] as usize;
            let col_max = col_maxs[di] as usize;
            if limb_offset >= col_max {
                continue;
            }
            let row_max = row_maxs[di] as usize;
            if row_max == 0 {
                continue;
            }
            let row_start = row_starts[di] as usize;
            let x_pm = &mut x_pm[..16 * row_max];
            unsafe {
                extract_blk_pair_prime_major_strided_avx2(n, row_max, bp, a_u64, a_cols, dsize - 1 - di, dsize, row_start, x_pm);
            }

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx2(
                        meta,
                        row_max,
                        blkpair_output,
                        x_pm,
                        &pmat_u64[y_off..],
                        plane_stride,
                    );
                    let base = col_res * 4 * n;
                    let blk0 = 2 * bp;
                    let blk1 = blk0 + 1;
                    if di == 0 {
                        save_blk_overwrite(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]);
                        save_blk_overwrite(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]);
                    } else {
                        save_blk_add(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]);
                        save_blk_add(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Q_SHIFTED, extract_1blk_from_contiguous_q120b_avx2, extract_blk_pair_prime_major_avx2,
        extract_blk_pair_prime_major_strided_avx2, save_blk_add,
    };
    use poulpy_cpu_ref::reference::ntt4x30::mat_vec::extract_1blk_from_contiguous_q120b_ref;

    #[test]
    fn extract_1blk_from_contiguous_q120b_avx2_vs_ref() {
        for &n in &[256usize, 4096, 16384] {
            for &row_max in &[1usize, 3, 7] {
                let src: Vec<u64> = (0..row_max * 4 * n)
                    .map(|i| (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1)) ^ ((i as u64) << 17))
                    .collect();

                for &blk in &[0usize, n / 4, n / 2 - 1] {
                    let mut dst_ref = vec![0u64; 8 * row_max];
                    let mut dst_avx = vec![0u64; 8 * row_max];

                    extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, &mut dst_ref, &src);
                    unsafe { extract_1blk_from_contiguous_q120b_avx2(n, row_max, blk, &mut dst_avx, &src) };

                    assert_eq!(dst_avx, dst_ref, "n={n}, row_max={row_max}, blk={blk}");
                }
            }
        }
    }

    #[test]
    fn extract_strided_digit_matches_materialized_avx2() {
        for &(n, cols, dsize, a_size) in &[(64usize, 1usize, 2usize, 4usize), (64, 2, 3, 8), (256, 2, 3, 7)] {
            let src: Vec<u64> = (0..cols * a_size * 4 * n)
                .map(|i| 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1))
                .collect();
            for di in 0..dsize {
                let digit_size = (a_size + di) / dsize;
                let limb_base = dsize - 1 - di;
                let mut materialized = Vec::with_capacity(cols * digit_size * 4 * n);
                for j in 0..digit_size {
                    for col in 0..cols {
                        let flat = (limb_base + j * dsize) * cols + col;
                        materialized.extend_from_slice(&src[flat * 4 * n..(flat + 1) * 4 * n]);
                    }
                }
                let row_max = cols * digit_size;
                for &bp in &[0usize, n / 8, n / 4 - 1] {
                    let mut expected = vec![0u64; 16 * row_max];
                    let mut got = vec![0u64; 16 * row_max];
                    unsafe { extract_blk_pair_prime_major_avx2(n, row_max, bp, &materialized, &mut expected) };
                    unsafe {
                        extract_blk_pair_prime_major_strided_avx2(n, row_max, bp, &src, cols, limb_base, dsize, 0, &mut got)
                    };
                    assert_eq!(
                        got, expected,
                        "n={n}, cols={cols}, dsize={dsize}, a_size={a_size}, di={di}, bp={bp}"
                    );
                }
            }
        }
    }

    #[test]
    fn save_blk_add_matches_scalar_modulo() {
        let n = 64usize;
        let blk = 3usize;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for iteration in 0..256 {
            let mut dst = vec![0u64; 4 * n];
            let mut src = [0u64; 8];
            for i in 0..8 {
                let q = Q_SHIFTED[i % 4];
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                dst[8 * blk + i] = state % (2 * q);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                src[i] = state % (2 * q);
            }

            // Include every reduction boundary explicitly as well as random inputs.
            if iteration == 0 {
                for i in 0..8 {
                    let q = Q_SHIFTED[i % 4];
                    dst[8 * blk + i] = [0, q - 1, q, 2 * q - 1][i % 4];
                    src[i] = [2 * q - 1, q, q - 1, 0][i % 4];
                }
            }

            let mut expected = dst.clone();
            for i in 0..8 {
                let q = Q_SHIFTED[i % 4];
                expected[8 * blk + i] = dst[8 * blk + i] % q + src[i] % q;
            }

            unsafe { save_blk_add(n, blk, &mut dst, &src) };
            assert_eq!(dst, expected, "iteration={iteration}");
        }
    }
}
