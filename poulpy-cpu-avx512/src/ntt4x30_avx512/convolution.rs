//! Fused lazy-canonicalization convolution apply for [`NTT4x30Avx512`], used by
//! `glwe_mul_plain`: packs each block on the fly and stores results non-temporally.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::_mm_sfence;

use poulpy_cpu_ref::reference::ntt4x30::{mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{CnvTVecLBackendRef, CnvTVecRBackendRef, Module, VecZnxDftBackendMut, ZnxView, ZnxViewMut};

use super::{
    arithmetic_avx512::{pack_left_1blk_x2_avx512, pack_right_1blk_x2_avx512},
    mat_vec_avx512::vec_mat1col_product_x2_bbc_avx512,
};
use crate::NTT4x30Avx512;

pub(crate) fn cnv_apply_tvec_to_dft_avx_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    16 * (a_size + b_size) * size_of::<u32>()
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cnv_apply_tvec_to_dft_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvTVecLBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &CnvTVecRBackendRef<'_, NTT4x30Avx512>,
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
        unsafe {
            pack_left_1blk_x2_avx512(a_tmp, &a_raw_u64[a_col_offset_u64..], a_size, a_row_stride_u64, blk);
            pack_right_1blk_x2_avx512(b_tmp, &b_raw_u32[b_col_offset_u32..], b_size, b_row_stride_u32, blk);
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            unsafe {
                vec_mat1col_product_x2_bbc_avx512::<true>(
                    meta,
                    ell,
                    &mut res_u64[8 * blk..8 * blk + 8],
                    &a_tmp[16 * a_start..],
                    &b_tmp[16 * b_start..],
                );
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j)
            .fill(poulpy_cpu_ref::reference::ntt4x30::types::CrtWord([0; 4]));
    }
    _mm_sfence();
}
