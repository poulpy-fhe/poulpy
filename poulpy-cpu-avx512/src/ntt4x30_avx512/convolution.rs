//! Fused lazy-canonicalization convolution apply for [`NTT4x30Avx512`], used by
//! `glwe_mul_plain`: packs each block on the fly and stores results non-temporally.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi32, _mm512_cmp_epi32_mask, _mm512_loadu_si512, _mm512_mask_sub_epi32, _mm512_storeu_si512,
};

use poulpy_cpu_ref::reference::ntt4x30::{
    NttSubAssign,
    mat_vec::BbcMeta,
    primes::{PrimeSet, Primes30},
    vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{CnvPVecLBackendRef, CnvPVecRBackendRef, Module, VecZnxDftBackendMut, ZnxView, ZnxViewMut};

use super::{
    arithmetic_avx512::{pack_left_1blk_x2_avx512, pack_right_1blk_x2_avx512},
    mat_vec_avx512::{vec_mat_tile4_bbc_canonical_avx512, vec_mat1col_product_x2_bbc_avx512},
};
use crate::NTT4x30Avx512;

pub(crate) fn cnv_apply_dft_lazy_avx_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    16 * (a_size + b_size) * size_of::<u32>()
}

pub(crate) fn cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    const GROUP: usize = 16;
    let min_size = res_size.min(a_size + b_size);
    (3 * 16 * (a_size + 2 * 3) + 16 * b_size) * size_of::<u32>() + 3 * 8 * GROUP * min_size * size_of::<u64>()
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cnv_apply_dft_lazy_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
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

#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cnv_tensor_rank1_dft_avx512(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    cnv_offset: usize,
    a: &CnvPVecLBackendRef<'_, NTT4x30Avx512>,
    b: &CnvPVecRBackendRef<'_, NTT4x30Avx512>,
    tmp: &mut [u8],
) {
    const TILE: usize = 4;
    const GROUP: usize = 16;
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
                cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
            }
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;
    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (stage, rest) = tmp_u64.split_at_mut(3 * 8 * GROUP * min_size);
    let rest_u32: &mut [u32] = cast_slice_mut(rest);
    let (a0_win, rest_u32) = rest_u32.split_at_mut(16 * win_rows);
    let (a1_win, rest_u32) = rest_u32.split_at_mut(16 * win_rows);
    let (a_sum_win, b_sum) = rest_u32.split_at_mut(16 * win_rows);
    for win in [&mut *a0_win, &mut *a1_win, &mut *a_sum_win] {
        win[..16 * pad].fill(0);
        win[16 * (a_size + pad)..].fill(0);
    }

    let a_raw: &[u32] = cast_slice(a.raw());
    let b_raw: &[u32] = cast_slice(b.raw());
    let a_stride = 8 * n * a_size;
    let b_stride = 8 * n * b_size;
    let (a0, a1) = (&a_raw[..a_stride], &a_raw[a_stride..2 * a_stride]);
    let (b0, b1) = (&b_raw[..b_stride], &b_raw[b_stride..2 * b_stride]);
    let n_blks = n / 2;
    let n_tiles = min_size.div_ceil(TILE);
    let meta: &BbcMeta<Primes30> = module.get_bbc_meta();
    let stage_col_stride = 8 * GROUP * min_size;
    let mut diag0 = [0u64; 8 * TILE];
    let mut cross = [0u64; 8 * TILE];
    let mut diag1 = [0u64; 8 * TILE];
    let q_rows = [
        Primes30::Q[0],
        Primes30::Q[0],
        Primes30::Q[1],
        Primes30::Q[1],
        Primes30::Q[2],
        Primes30::Q[2],
        Primes30::Q[3],
        Primes30::Q[3],
        Primes30::Q[0],
        Primes30::Q[0],
        Primes30::Q[1],
        Primes30::Q[1],
        Primes30::Q[2],
        Primes30::Q[2],
        Primes30::Q[3],
        Primes30::Q[3],
    ];
    let q_vec = unsafe { _mm512_loadu_si512(q_rows.as_ptr() as *const __m512i) };

    for blk in 0..n_blks {
        let a0_blk = &a0[blk * 16 * a_size..(blk + 1) * 16 * a_size];
        let a1_blk = &a1[blk * 16 * a_size..(blk + 1) * 16 * a_size];
        a0_win[16 * pad..16 * (pad + a_size)].copy_from_slice(a0_blk);
        a1_win[16 * pad..16 * (pad + a_size)].copy_from_slice(a1_blk);
        for r in 0..a_size {
            unsafe {
                let x = _mm512_loadu_si512(a0_blk.as_ptr().add(16 * r) as *const __m512i);
                let y = _mm512_loadu_si512(a1_blk.as_ptr().add(16 * r) as *const __m512i);
                let sum = _mm512_add_epi32(x, y);
                let reduced = _mm512_mask_sub_epi32(sum, _mm512_cmp_epi32_mask::<5>(sum, q_vec), sum, q_vec);
                _mm512_storeu_si512(a_sum_win.as_mut_ptr().add(16 * (pad + r)) as *mut __m512i, reduced);
            }
        }

        let b0_blk = &b0[blk * 16 * b_size..(blk + 1) * 16 * b_size];
        let b1_blk = &b1[blk * 16 * b_size..(blk + 1) * 16 * b_size];
        for q in (0..16 * b_size).step_by(16) {
            unsafe {
                let x = _mm512_loadu_si512(b0_blk.as_ptr().add(q) as *const __m512i);
                let y = _mm512_loadu_si512(b1_blk.as_ptr().add(q) as *const __m512i);
                _mm512_storeu_si512(b_sum.as_mut_ptr().add(q) as *mut __m512i, _mm512_add_epi32(x, y));
            }
        }

        let group_pos = blk % GROUP;
        for tile in 0..n_tiles {
            let k0 = offset + TILE * tile;
            let j_lo = (k0 + 1).saturating_sub(a_size).min(b_size);
            let j_hi = (k0 + TILE).min(b_size);
            let len = j_hi.saturating_sub(j_lo);
            let win_base = (k0 + pad + 1)
                .saturating_sub(j_hi)
                .min(win_rows.saturating_sub(TILE - 1 + len));
            let b_start = b_size - j_hi;
            unsafe {
                vec_mat_tile4_bbc_canonical_avx512(meta, len, &mut diag0, &a0_win[16 * win_base..], &b0_blk[16 * b_start..]);
                vec_mat_tile4_bbc_canonical_avx512(meta, len, &mut cross, &a_sum_win[16 * win_base..], &b_sum[16 * b_start..]);
                vec_mat_tile4_bbc_canonical_avx512(meta, len, &mut diag1, &a1_win[16 * win_base..], &b1_blk[16 * b_start..]);
            }
            <NTT4x30Avx512 as NttSubAssign>::ntt_sub_assign(&mut cross, &diag0);
            <NTT4x30Avx512 as NttSubAssign>::ntt_sub_assign(&mut cross, &diag1);

            let k_rel = TILE * tile;
            for t in 0..TILE.min(min_size - k_rel) {
                for (col, values) in [(0, &diag0), (1, &cross), (2, &diag1)] {
                    let dst = col * stage_col_stride + 8 * ((k_rel + t) * GROUP + group_pos);
                    stage[dst..dst + 8].copy_from_slice(&values[8 * t..8 * (t + 1)]);
                }
            }
        }

        let in_group = group_pos + 1;
        if in_group == GROUP || blk == n_blks - 1 {
            let group_base = blk + 1 - in_group;
            for k in 0..min_size {
                for col in 0..3 {
                    let src = col * stage_col_stride + 8 * k * GROUP;
                    let dst: &mut [u64] = cast_slice_mut(res.at_mut(col, k));
                    dst[8 * group_base..8 * (group_base + in_group)].copy_from_slice(&stage[src..src + 8 * in_group]);
                }
            }
        }
    }

    for col in 0..3 {
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}
