//! Polynomial convolution NEON kernels for [`NTT120Neon`](crate::NTT120Neon).
//!
//! Reuses the block-outer pack-then-multiply structure from the generic NTT120
//! implementation, with `stnp q, q` non-temporal stores for the result so
//! convolution output lines do not evict the packed operand rows from cache.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::reference::ntt120::{mat_vec::BbcMeta, primes::Primes30, types::Q120bScalar, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{CnvPVecLBackendRef, CnvPVecRBackendRef, Module, VecZnxDftBackendMut, ZnxView, ZnxViewMut};

use super::super::neon::ntt120_convert::{
    pack_left_1blk_x2_neon, pack_right_1blk_x2_neon, pairwise_pack_left_1blk_x2_neon, pairwise_pack_right_1blk_x2_neon,
};
use super::super::neon::ntt120_mat_vec::vec_mat1col_product_x2_bbc_neon;
use crate::NTT120Neon;

pub(crate) fn cnv_apply_dft_neon_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    16 * (a_size + b_size) * size_of::<u32>()
}

pub(crate) fn cnv_pairwise_apply_dft_neon_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        (16 * (a_size + b_size) * size_of::<u32>()).max(cnv_apply_dft_neon_tmp_bytes(a_size, b_size))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_apply_dft_neon(
    module: &Module<NTT120Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT120Neon>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT120Neon>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT120Neon>,
    b_col: usize,
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let meta: &BbcMeta<Primes30> = module.get_bbc_meta();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let a_row_stride_u64 = 4 * n * a_cols;
    let b_row_stride_u32 = 8 * n * b_cols;
    let a_col_offset_u64 = 4 * n * a_col;
    let b_col_offset_u32 = 8 * n * b_col;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u32: &[u32] = cast_slice(b.raw());

    let (prefix, tmp_u32, suffix) = unsafe { tmp.align_to_mut::<u32>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u32.len() >= 16 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u32.split_at_mut(16 * a_size);

    for blk in 0..n_blks {
        pack_left_1blk_x2_neon(a_tmp, &a_raw_u64[a_col_offset_u64..], a_size, a_row_stride_u64, blk);
        pack_right_1blk_x2_neon(b_tmp, &b_raw_u32[b_col_offset_u32..], b_size, b_row_stride_u32, blk);

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            let ell = j_max - k_abs.saturating_sub(a_size - 1);

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            vec_mat1col_product_x2_bbc_neon::<true>(
                meta,
                ell,
                &mut res_u64[8 * blk..8 * blk + 8],
                &a_tmp[16 * a_start..],
                &b_tmp[16 * b_start..],
            );
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_pairwise_apply_dft_neon(
    module: &Module<NTT120Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT120Neon>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT120Neon>,
    b: &CnvPVecRBackendRef<'_, NTT120Neon>,
    col_i: usize,
    col_j: usize,
    tmp: &mut [u8],
) {
    if col_i == col_j {
        cnv_apply_dft_neon(module, res, cnv_offset, res_col, a, col_i, b, col_j, tmp);
        return;
    }

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let meta: &BbcMeta<Primes30> = module.get_bbc_meta();
    let a_cols = a.cols();
    let b_cols = b.cols();

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let n_blks = n / 2;
    let a_row_stride_u64 = 4 * n * a_cols;
    let b_row_stride_u32 = 8 * n * b_cols;
    let a_col_offset_u64_i = 4 * n * col_i;
    let a_col_offset_u64_j = 4 * n * col_j;
    let b_col_offset_u32_i = 8 * n * col_i;
    let b_col_offset_u32_j = 8 * n * col_j;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u32: &[u32] = cast_slice(b.raw());

    let (prefix, tmp_u32, suffix) = unsafe { tmp.align_to_mut::<u32>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u32.len() >= 16 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u32.split_at_mut(16 * a_size);

    for blk in 0..n_blks {
        pairwise_pack_left_1blk_x2_neon(
            a_tmp,
            &a_raw_u64[a_col_offset_u64_i..],
            &a_raw_u64[a_col_offset_u64_j..],
            a_size,
            a_row_stride_u64,
            blk,
        );
        pairwise_pack_right_1blk_x2_neon(
            b_tmp,
            &b_raw_u32[b_col_offset_u32_i..],
            &b_raw_u32[b_col_offset_u32_j..],
            b_size,
            b_row_stride_u32,
            blk,
        );

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_max = (k_abs + 1).min(b_size);
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;
            let ell = j_max - k_abs.saturating_sub(a_size - 1);

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            vec_mat1col_product_x2_bbc_neon::<true>(
                meta,
                ell,
                &mut res_u64[8 * blk..8 * blk + 8],
                &a_tmp[16 * a_start..],
                &b_tmp[16 * b_start..],
            );
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
}
