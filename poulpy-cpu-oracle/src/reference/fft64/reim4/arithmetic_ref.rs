// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

use crate::reference::fft64::reim::{as_arr, as_arr_mut, reim_zero_ref};

#[inline(always)]
pub fn reim4_extract_1blk_from_reim_contiguous_ref(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    assert!(blk < (m >> 2));
    assert!(dst.len() >= 2 * rows * 4);

    let offset: usize = blk << 2;

    // src = 4-values chunks spaced by m, dst = sequential 4-values chunks
    let src_rows = src.chunks_exact(m).take(2 * rows);
    let dst_chunks = dst.chunks_exact_mut(4).take(2 * rows);

    for (dst_chunk, src_row) in dst_chunks.zip(src_rows) {
        dst_chunk.copy_from_slice(&src_row[offset..offset + 4]);
    }
}

#[inline(always)]
pub fn reim4_save_1blk_to_reim_contiguous_ref(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    assert!(blk < (m >> 2));
    assert!(src.len() >= 2 * rows * 4);

    let offset: usize = blk << 2;

    // dst = 4-values chunks spaced by m, src = sequential 4-values chunks
    let dst_rows = dst.chunks_exact_mut(m).take(2 * rows);
    let src_chunks = src.chunks_exact(4).take(2 * rows);

    for (dst_row, src_chunk) in dst_rows.zip(src_chunks) {
        dst_row[offset..offset + 4].copy_from_slice(src_chunk);
    }
}

#[inline(always)]
pub fn reim4_save_1blk_to_reim_ref<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    assert!(blk < (m >> 2));
    assert!(dst.len() >= offset + m + 4);
    assert!(src.len() >= 8);

    let dst_off = &mut dst[offset..offset + 4];

    if OVERWRITE {
        dst_off.copy_from_slice(&src[0..4]);
    } else {
        dst_off[0] += src[0];
        dst_off[1] += src[1];
        dst_off[2] += src[2];
        dst_off[3] += src[3];
    }

    offset += m;

    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[4..8]);
    } else {
        dst_off[0] += src[4];
        dst_off[1] += src[5];
        dst_off[2] += src[6];
        dst_off[3] += src[7];
    }
}

#[inline(always)]
pub fn reim4_save_2blk_to_reim_ref<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    assert!(blk < (m >> 2));
    assert!(dst.len() >= offset + 3 * m + 4);
    assert!(src.len() >= 16);

    let dst_off: &mut [f64] = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[0..4]);
    } else {
        dst_off[0] += src[0];
        dst_off[1] += src[1];
        dst_off[2] += src[2];
        dst_off[3] += src[3];
    }

    offset += m;
    let dst_off: &mut [f64] = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[4..8]);
    } else {
        dst_off[0] += src[4];
        dst_off[1] += src[5];
        dst_off[2] += src[6];
        dst_off[3] += src[7];
    }

    offset += m;

    let dst_off: &mut [f64] = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[8..12]);
    } else {
        dst_off[0] += src[8];
        dst_off[1] += src[9];
        dst_off[2] += src[10];
        dst_off[3] += src[11];
    }

    offset += m;
    let dst_off: &mut [f64] = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[12..16]);
    } else {
        dst_off[0] += src[12];
        dst_off[1] += src[13];
        dst_off[2] += src[14];
        dst_off[3] += src[15];
    }
}

#[inline(always)]
pub fn reim4_vec_mat1col_product_ref(
    nrows: usize,
    dst: &mut [f64], // 8 doubles: [re1(4), im1(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 8 doubles: [ar(4) | ai(4)] per row
) {
    {
        assert!(dst.len() >= 8, "dst must have at least 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
    }

    let mut acc: [f64; 8] = [0f64; 8];
    let mut j = 0;
    for _ in 0..nrows {
        reim4_add_mul(&mut acc, as_arr(&u[j..]), as_arr(&v[j..]));
        j += 8;
    }
    dst[0..8].copy_from_slice(&acc);
}

#[inline(always)]
pub fn reim4_vec_mat2cols_product_ref(
    nrows: usize,
    dst: &mut [f64], // 16 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [ar(4) | ai(4) | br(4) | bi(4)] per row
) {
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 16, "v must be at least nrows * 16 doubles");
    }

    // zero accumulators
    let mut acc_0: [f64; 8] = [0f64; 8];
    let mut acc_1: [f64; 8] = [0f64; 8];
    for i in 0..nrows {
        let _1j: usize = i << 3;
        let _2j: usize = i << 4;
        let u_j: &[f64; 8] = as_arr(&u[_1j..]);
        reim4_add_mul(&mut acc_0, u_j, as_arr(&v[_2j..]));
        reim4_add_mul(&mut acc_1, u_j, as_arr(&v[_2j + 8..]));
    }
    dst[0..8].copy_from_slice(&acc_0);
    dst[8..16].copy_from_slice(&acc_1);
}

#[inline(always)]
pub fn reim4_vec_mat2cols_2ndcol_product_ref(
    nrows: usize,
    dst: &mut [f64], // 8 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [x | x | br(4) | bi(4)] per row
) {
    {
        assert!(dst.len() >= 8, "dst must be at least 8 doubles but is {}", dst.len());
        assert!(
            u.len() >= nrows * 8,
            "u must be at least nrows={} * 8 doubles but is {}",
            nrows,
            u.len()
        );
        assert!(
            v.len() >= nrows * 16,
            "v must be at least nrows={} * 16 doubles but is {}",
            nrows,
            v.len()
        );
    }

    // zero accumulators
    let mut acc: [f64; 8] = [0f64; 8];
    for i in 0..nrows {
        let _1j: usize = i << 3;
        let _2j: usize = i << 4;
        reim4_add_mul(&mut acc, as_arr(&u[_1j..]), as_arr(&v[_2j + 8..]));
    }
    dst[0..8].copy_from_slice(&acc);
}

#[inline(always)]
pub fn reim4_add_mul(dst: &mut [f64; 8], a: &[f64; 8], b: &[f64; 8]) {
    for k in 0..4 {
        let ar: f64 = a[k];
        let br: f64 = b[k];
        let ai: f64 = a[k + 4];
        let bi: f64 = b[k + 4];
        dst[k] += ar * br - ai * bi;
        dst[k + 4] += ar * bi + ai * br;
    }
}

#[inline(always)]
pub fn reim4_convolution_1coeff_ref(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    reim_zero_ref(dst);

    if k >= a_size + b_size {
        return;
    }
    let j_min: usize = k.saturating_sub(a_size - 1);
    let j_max: usize = (k + 1).min(b_size);

    for j in j_min..j_max {
        reim4_add_mul(dst, as_arr(&a[8 * (k - j)..]), as_arr(&b[8 * j..]));
    }
}

#[inline(always)]
pub fn reim4_convolution_2coeffs_ref(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    reim4_convolution_1coeff_ref(k, as_arr_mut(dst), a, a_size, b, b_size);
    reim4_convolution_1coeff_ref(k + 1, as_arr_mut(&mut dst[8..]), a, a_size, b, b_size);
}

#[inline(always)]
pub fn reim4_add_mul_b_real_const(dst: &mut [f64; 8], a: &[f64; 8], b: f64) {
    for k in 0..4 {
        let ar: f64 = a[k];
        let ai: f64 = a[k + 4];
        dst[k] += ar * b;
        dst[k + 4] += ai * b;
    }
}

#[inline(always)]
pub fn reim4_convolution_by_real_const_1coeff_ref(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
    reim_zero_ref(dst);

    let b_size: usize = b.len();

    if k >= a_size + b_size {
        return;
    }
    let j_min: usize = k.saturating_sub(a_size - 1);
    let j_max: usize = (k + 1).min(b_size);

    for j in j_min..j_max {
        reim4_add_mul_b_real_const(dst, as_arr(&a[8 * (k - j)..]), b[j]);
    }
}

#[inline(always)]
pub fn reim4_convolution_by_real_const_2coeffs_ref(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
    reim4_convolution_by_real_const_1coeff_ref(k, as_arr_mut(dst), a, a_size, b);
    reim4_convolution_by_real_const_1coeff_ref(k + 1, as_arr_mut(&mut dst[8..]), a, a_size, b);
}
