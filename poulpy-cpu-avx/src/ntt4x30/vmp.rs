//! Vector-matrix product AVX2 kernels for [`NTT4x30Avx`](crate::NTT4x30Avx).
//!
//! Uses a backend-local prime-major prepared-matrix layout so the hot AVX VMP
//! path streams one prime plane at a time and reuses extracted input rows
//! across the output-column loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_set_epi64x, _mm256_storeu_si256,
};

use poulpy_core::oep::gglwe_product_digit_output_size;
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
    VmpPMatBackendRef, ZnxView, ZnxViewMut,
};

use super::arithmetic_avx::{BARRETT_MU, POW32, Q_VEC, cond_sub, reduce_b_to_canonical};
use super::mat_vec_avx::vec_mat1col_product_blkpair_bbc_pm_avx2;
use super::vec_znx_dft::{canonicalize_limb_q120, pack_two_q120};
use crate::NTT4x30Avx;

#[derive(Clone, Copy)]
struct SendU32Ptr(*mut u32);

// SAFETY: this pointer is only shared while each task owns a distinct NTT
// block pair. Callers join all tasks before accessing the output again.
unsafe impl Send for SendU32Ptr {}
unsafe impl Sync for SendU32Ptr {}

impl SendU32Ptr {
    #[inline(always)]
    fn get(&self) -> *mut u32 {
        self.0
    }
}

/// Scratch space (in bytes) required by the AVX VMP prepare kernel.
pub(crate) fn vmp_prepare_tmp_bytes_avx(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// AVX-local VMP prepare into two packed prime-pair planes.
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
    let pair_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    let tmp_b = &mut tmp[..4 * n];

    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            NTT4x30Avx::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT4x30Avx::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            unsafe { canonicalize_limb_q120(n, tmp_b) };

            for bp in 0..n_block_pairs {
                let coeff_base = 16 * bp;
                for pair in 0..2usize {
                    let p = 2 * pair;
                    let dst = pair * pair_stride + bp * bp_stride + col_i * col_stride + row_i * 4;
                    for coeff in 0..4 {
                        let coeff_off = coeff_base + 4 * coeff;
                        pmat_u64[dst + coeff] = tmp_b[coeff_off + p] | (tmp_b[coeff_off + p + 1] << 32);
                    }
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
unsafe fn extract_blk_pair_prime_major_avx2(n: usize, row_max: usize, blk_pair: usize, src: &[u32], dst: &mut [u64]) {
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
    src: &[u32],
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

#[target_feature(enable = "avx2")]
unsafe fn save_blk_overwrite(blk: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
        let a = reduce_b_to_canonical(_mm256_loadu_si256(src.as_ptr() as *const __m256i), q, mu, pow32);
        let b = reduce_b_to_canonical(_mm256_loadu_si256(src.as_ptr().add(4) as *const __m256i), q, mu, pow32);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, pack_two_q120(a, b));
    }
}

#[target_feature(enable = "avx2")]
unsafe fn save_blk_add(blk: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 8 * (blk + 1));
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
        let dst_ptr = dst.as_mut_ptr().add(8 * blk) as *mut __m256i;
        let packed = _mm256_loadu_si256(dst_ptr);
        let d0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(packed));
        let d1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(packed));
        let s0 = reduce_b_to_canonical(_mm256_loadu_si256(src.as_ptr() as *const __m256i), q, mu, pow32);
        let s1 = reduce_b_to_canonical(_mm256_loadu_si256(src.as_ptr().add(4) as *const __m256i), q, mu, pow32);
        let r0 = cond_sub(_mm256_add_epi64(d0, s0), q);
        let r1 = cond_sub(_mm256_add_epi64(d1, s1), q);
        _mm256_storeu_si256(dst_ptr, pack_two_q120(r0, r1));
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn vmp_apply_core_avx_pm<const OVERWRITE: bool, E: TaskExecutor>(
    n: usize,
    res_u32: &mut [u32],
    a_u32: &[u32],
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

    let a_size = a_u32.len() / (4 * n);
    let res_size = res_u32.len() / (4 * n);
    let n_block_pairs = n / 4;

    let row_end = nrows.min(a_size);
    let row_start = a_u32
        .chunks_exact(4 * n)
        .take(row_end)
        .take_while(|row| row.iter().all(|&x| x == 0))
        .count();
    let row_max = row_end - row_start;
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max || row_max == 0 {
        if OVERWRITE {
            res_u32.fill(0);
        }
        return;
    }

    let pair_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;
    let a_u32 = &a_u32[row_start * 4 * n..];

    if !E::is_parallel() || n_block_pairs < 2 {
        let (blkpair_output, x_pm) = tmp.split_at_mut(16);
        let x_pm = &mut x_pm[..16 * row_max];
        for bp in 0..n_block_pairs {
            unsafe { extract_blk_pair_prime_major_avx2(n, row_max, bp, a_u32, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;

                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx2(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], pair_stride)
                };

                let blk0 = 2 * bp;
                let blk1 = blk0 + 1;
                let base = col_res * 4 * n;
                if OVERWRITE {
                    unsafe { save_blk_overwrite(blk0, &mut res_u32[base..], &blkpair_output[0..8]) };
                    unsafe { save_blk_overwrite(blk1, &mut res_u32[base..], &blkpair_output[8..16]) };
                } else {
                    unsafe { save_blk_add(blk0, &mut res_u32[base..], &blkpair_output[0..8]) };
                    unsafe { save_blk_add(blk1, &mut res_u32[base..], &blkpair_output[8..16]) };
                }
            }
        }
    } else {
        let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
        E::for_each_chunked(n_block_pairs, tmp, 16 + 16 * row_max, |task_tmp, bp| {
            let (blkpair_output, x_pm) = task_tmp.split_at_mut(16);
            unsafe { extract_blk_pair_prime_major_avx2(n, row_max, bp, a_u32, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx2(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], pair_stride);
                    let base = col_res * 4 * n;
                    let blk0 = 2 * bp;
                    let blk1 = blk0 + 1;
                    let dst0 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 8 * blk0), 8);
                    let dst1 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 8 * blk1), 8);
                    if OVERWRITE {
                        save_blk_overwrite(0, dst0, &blkpair_output[0..8]);
                        save_blk_overwrite(0, dst1, &blkpair_output[8..16]);
                    } else {
                        save_blk_add(0, dst0, &blkpair_output[0..8]);
                        save_blk_add(0, dst1, &blkpair_output[8..16]);
                    }
                }
            }
        });
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u32[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

pub(crate) fn vmp_apply_dft_to_dft_avx<E: TaskExecutor>(
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

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_avx_pm::<true, E>(
            n,
            res_u32,
            a_u32,
            pmat_u64,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}

pub(crate) fn vmp_apply_dft_to_dft_accumulate_avx<E: TaskExecutor>(
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

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_avx_pm::<false, E>(
            n,
            res_u32,
            a_u32,
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
    workers: usize,
) -> usize {
    let nrows = b_rows * b_cols_in;
    let row_max = (0..dsize)
        .map(|di| (a_cols * ((a_size + di) / dsize).min(b_rows)).min(nrows))
        .max()
        .unwrap_or(0);
    (4 * dsize + workers * (16 + 16 * row_max)) * size_of::<u64>()
}

/// Applies all gadget digits directly from their interleaved source limbs.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_avx<E: TaskExecutor>(
    module: &Module<NTT4x30Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx>,
    dsize: usize,
    product_limbs: usize,
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
    let a_u32: &[u32] = cast_slice(a.raw());
    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let pair_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    let (digit_meta, tmp) = tmp.split_at_mut(4 * dsize);
    let (row_maxs, digit_meta) = digit_meta.split_at_mut(dsize);
    let (row_starts, digit_meta) = digit_meta.split_at_mut(dsize);
    let (limb_offsets, col_maxs) = digit_meta.split_at_mut(dsize);
    for di in 0..dsize {
        let digit_limbs = ((a_size + di) / dsize).min(dnum);
        // Match the reference product: full-width overwrite, then narrowed accumulations.
        let active_size = gglwe_product_digit_output_size(output_size, pmat.size(), dsize, di, product_limbs);
        let limb_offset = di * cols_out;
        let row_end = (a_cols * digit_limbs).min(nrows);
        let limb_base = dsize - 1 - di;
        let row_start = (0..row_end)
            .take_while(|&row| {
                let flat = (limb_base + (row / a_cols) * dsize) * a_cols + row % a_cols;
                a_u32[flat * 4 * n..(flat + 1) * 4 * n].iter().all(|&x| x == 0)
            })
            .count();
        row_starts[di] = row_start as u64;
        row_maxs[di] = (row_end - row_start) as u64;
        limb_offsets[di] = limb_offset as u64;
        col_maxs[di] = ncols.min(res_cols * active_size + limb_offset) as u64;
    }

    let res_flat = res_cols * output_size;
    if row_maxs[0] == 0 {
        res_u32.fill(0);
    } else {
        for col in col_maxs[0] as usize..res_flat {
            res_u32[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }

    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
    let process_block_pair = |task_tmp: &mut [u64], bp: usize| {
        let (blkpair_output, x_pm) = task_tmp.split_at_mut(16);
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
                extract_blk_pair_prime_major_strided_avx2(n, row_max, bp, a_u32, a_cols, dsize - 1 - di, dsize, row_start, x_pm);
            }

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx2(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], pair_stride);
                    let base = col_res * 4 * n;
                    let blk0 = 2 * bp;
                    let blk1 = blk0 + 1;
                    let dst0 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 8 * blk0), 8);
                    let dst1 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 8 * blk1), 8);
                    if di == 0 {
                        save_blk_overwrite(0, dst0, &blkpair_output[0..8]);
                        save_blk_overwrite(0, dst1, &blkpair_output[8..16]);
                    } else {
                        save_blk_add(0, dst0, &blkpair_output[0..8]);
                        save_blk_add(0, dst1, &blkpair_output[8..16]);
                    }
                }
            }
        }
    };

    if E::is_parallel() && n_block_pairs > 1 {
        let max_row_max = row_maxs.iter().copied().max().unwrap_or(0) as usize;
        E::for_each_chunked(n_block_pairs, tmp, 16 + 16 * max_row_max, process_block_pair);
    } else {
        for bp in 0..n_block_pairs {
            process_block_pair(tmp, bp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::extract_1blk_from_contiguous_q120b_avx2;
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
}
