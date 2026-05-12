//! Vector-matrix product AVX2 kernels for [`NTT120Avx`](crate::NTT120Avx).
//!
//! Uses a backend-local prime-major prepared-matrix layout so the hot AVX VMP
//! path streams one prime plane at a time and reuses extracted input rows
//! across the output-column loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{__m256i, _mm_sfence, _mm256_loadu_si256, _mm256_set_epi64x, _mm256_storeu_si256, _mm256_stream_si256};

use poulpy_cpu_ref::reference::ntt120::{
    NttCFromB, NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, types::Q_SHIFTED, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{
    DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    ZnxView, ZnxViewMut,
};

use super::mat_vec_avx::vec_mat1col_product_blkpair_bbc_pm_avx2;
use crate::NTT120Avx;

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
    module: &Module<NTT120Avx>,
    res: &mut VmpPMatBackendMut<'_, NTT120Avx>,
    a: &MatZnxBackendRef<'_, NTT120Avx>,
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

            NTT120Avx::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT120Avx::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            NTT120Avx::ntt_c_from_b(n, tmp_c, tmp_b);
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

/// Non-temporal write of one x2-block (8 u64) into a q120b vector.
///
/// `VecZnxDft` storage is 64-byte aligned, and every x2-block offset is a
/// multiple of 64 bytes, so both 256-bit stream stores land on aligned cache
/// line halves.
#[target_feature(enable = "avx2")]
unsafe fn save_blk_overwrite_nt(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m256i };
    let src_ptr = src.as_ptr() as *const __m256i;
    unsafe {
        _mm256_stream_si256(dst_ptr, _mm256_loadu_si256(src_ptr));
        _mm256_stream_si256(dst_ptr.add(1), _mm256_loadu_si256(src_ptr.add(1)));
    }
}

#[inline(always)]
fn save_blk_add(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    for i in 0..8 {
        let k = i % 4;
        dst[8 * blk + i] = dst[8 * blk + i] % Q_SHIFTED[k] + src[i] % Q_SHIFTED[k];
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

    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
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

    for bp in 0..n_block_pairs {
        unsafe { extract_blk_pair_prime_major_avx2(n, row_max, bp, a_u64, x_pm) };

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bp * bp_stride + col_pmat * col_stride;

            unsafe {
                vec_mat1col_product_blkpair_bbc_pm_avx2(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], plane_stride)
            };

            let blk0 = 2 * bp;
            let blk1 = blk0 + 1;
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite_nt(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]) };
                unsafe { save_blk_overwrite_nt(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]) };
            } else {
                save_blk_add(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]);
                save_blk_add(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]);
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        _mm_sfence();
    }
}

pub(crate) fn vmp_apply_dft_to_dft_avx(
    module: &Module<NTT120Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT120Avx>,
    a: &VecZnxDftBackendRef<'_, NTT120Avx>,
    pmat: &VmpPMatBackendRef<'_, NTT120Avx>,
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
    module: &Module<NTT120Avx>,
    res: &mut VecZnxDftBackendMut<'_, NTT120Avx>,
    a: &VecZnxDftBackendRef<'_, NTT120Avx>,
    pmat: &VmpPMatBackendRef<'_, NTT120Avx>,
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

#[cfg(test)]
mod tests {
    use super::extract_1blk_from_contiguous_q120b_avx2;
    use poulpy_cpu_ref::reference::ntt120::mat_vec::extract_1blk_from_contiguous_q120b_ref;

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
