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

use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use crate::reference::fft64::reim::{as_arr, as_arr_mut};

#[inline(always)]
pub fn fft_ref<R: Float + FloatConst + Debug>(m: usize, omg: &[R], data: &mut [R]) {
    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m <= 16 {
        match m {
            1 => {}
            2 => fft2_ref(as_arr_mut::<2, R>(re), as_arr_mut::<2, R>(im), *as_arr::<2, R>(omg)),
            4 => fft4_ref(as_arr_mut::<4, R>(re), as_arr_mut::<4, R>(im), *as_arr::<4, R>(omg)),
            8 => fft8_ref(as_arr_mut::<8, R>(re), as_arr_mut::<8, R>(im), *as_arr::<8, R>(omg)),
            16 => fft16_ref(as_arr_mut::<16, R>(re), as_arr_mut::<16, R>(im), *as_arr::<16, R>(omg)),
            _ => {}
        }
    } else if m <= 2048 {
        fft_bfs_16_ref(m, re, im, omg, 0);
    } else {
        fft_rec_16_ref(m, re, im, omg, 0);
    }
}

#[inline(always)]
fn fft_rec_16_ref<R: Float + FloatConst + Debug>(m: usize, re: &mut [R], im: &mut [R], omg: &[R], mut pos: usize) -> usize {
    if m <= 2048 {
        return fft_bfs_16_ref(m, re, im, omg, pos);
    };

    let h = m >> 1;
    twiddle_fft_ref(h, re, im, as_arr::<2, R>(&omg[pos..]));
    pos += 2;
    pos = fft_rec_16_ref(h, re, im, omg, pos);
    pos = fft_rec_16_ref(h, &mut re[h..], &mut im[h..], omg, pos);
    pos
}

#[inline(always)]
fn cplx_twiddle<R: Float + FloatConst>(ra: &mut R, ia: &mut R, rb: &mut R, ib: &mut R, omg_re: R, omg_im: R) {
    let dr: R = *rb * omg_re - *ib * omg_im;
    let di: R = *rb * omg_im + *ib * omg_re;
    *rb = *ra - dr;
    *ib = *ia - di;
    *ra = *ra + dr;
    *ia = *ia + di;
}

#[inline(always)]
fn cplx_i_twiddle<R: Float + FloatConst>(ra: &mut R, ia: &mut R, rb: &mut R, ib: &mut R, omg_re: R, omg_im: R) {
    let dr: R = *rb * omg_im + *ib * omg_re;
    let di: R = *rb * omg_re - *ib * omg_im;
    *rb = *ra + dr;
    *ib = *ia - di;
    *ra = *ra - dr;
    *ia = *ia + di;
}

#[inline(always)]
fn fft2_ref<R: Float + FloatConst>(re: &mut [R; 2], im: &mut [R; 2], omg: [R; 2]) {
    let [ra, rb] = re;
    let [ia, ib] = im;
    let [romg, iomg] = omg;
    cplx_twiddle(ra, ia, rb, ib, romg, iomg);
}

#[inline(always)]
fn fft4_ref<R: Float + FloatConst>(re: &mut [R; 4], im: &mut [R; 4], omg: [R; 4]) {
    let [re_0, re_1, re_2, re_3] = re;
    let [im_0, im_1, im_2, im_3] = im;

    {
        let omg_0 = omg[0];
        let omg_1 = omg[1];
        cplx_twiddle(re_0, im_0, re_2, im_2, omg_0, omg_1);
        cplx_twiddle(re_1, im_1, re_3, im_3, omg_0, omg_1);
    }

    {
        let omg_0 = omg[2];
        let omg_1 = omg[3];
        cplx_twiddle(re_0, im_0, re_1, im_1, omg_0, omg_1);
        cplx_i_twiddle(re_2, im_2, re_3, im_3, omg_0, omg_1);
    }
}

#[inline(always)]
fn fft8_ref<R: Float + FloatConst>(re: &mut [R; 8], im: &mut [R; 8], omg: [R; 8]) {
    let [re_0, re_1, re_2, re_3, re_4, re_5, re_6, re_7] = re;
    let [im_0, im_1, im_2, im_3, im_4, im_5, im_6, im_7] = im;

    {
        let omg_0 = omg[0];
        let omg_1 = omg[1];
        cplx_twiddle(re_0, im_0, re_4, im_4, omg_0, omg_1);
        cplx_twiddle(re_1, im_1, re_5, im_5, omg_0, omg_1);
        cplx_twiddle(re_2, im_2, re_6, im_6, omg_0, omg_1);
        cplx_twiddle(re_3, im_3, re_7, im_7, omg_0, omg_1);
    }

    {
        let omg_2 = omg[2];
        let omg_3 = omg[3];
        cplx_twiddle(re_0, im_0, re_2, im_2, omg_2, omg_3);
        cplx_twiddle(re_1, im_1, re_3, im_3, omg_2, omg_3);
        cplx_i_twiddle(re_4, im_4, re_6, im_6, omg_2, omg_3);
        cplx_i_twiddle(re_5, im_5, re_7, im_7, omg_2, omg_3);
    }

    {
        let omg_4 = omg[4];
        let omg_5 = omg[5];
        let omg_6 = omg[6];
        let omg_7 = omg[7];
        cplx_twiddle(re_0, im_0, re_1, im_1, omg_4, omg_6);
        cplx_i_twiddle(re_2, im_2, re_3, im_3, omg_4, omg_6);
        cplx_twiddle(re_4, im_4, re_5, im_5, omg_5, omg_7);
        cplx_i_twiddle(re_6, im_6, re_7, im_7, omg_5, omg_7);
    }
}

#[inline(always)]
fn fft16_ref<R: Float + FloatConst + Debug>(re: &mut [R; 16], im: &mut [R; 16], omg: [R; 16]) {
    let [
        re_0,
        re_1,
        re_2,
        re_3,
        re_4,
        re_5,
        re_6,
        re_7,
        re_8,
        re_9,
        re_10,
        re_11,
        re_12,
        re_13,
        re_14,
        re_15,
    ] = re;
    let [
        im_0,
        im_1,
        im_2,
        im_3,
        im_4,
        im_5,
        im_6,
        im_7,
        im_8,
        im_9,
        im_10,
        im_11,
        im_12,
        im_13,
        im_14,
        im_15,
    ] = im;

    {
        let omg_0: R = omg[0];
        let omg_1: R = omg[1];
        cplx_twiddle(re_0, im_0, re_8, im_8, omg_0, omg_1);
        cplx_twiddle(re_1, im_1, re_9, im_9, omg_0, omg_1);
        cplx_twiddle(re_2, im_2, re_10, im_10, omg_0, omg_1);
        cplx_twiddle(re_3, im_3, re_11, im_11, omg_0, omg_1);

        cplx_twiddle(re_4, im_4, re_12, im_12, omg_0, omg_1);
        cplx_twiddle(re_5, im_5, re_13, im_13, omg_0, omg_1);
        cplx_twiddle(re_6, im_6, re_14, im_14, omg_0, omg_1);
        cplx_twiddle(re_7, im_7, re_15, im_15, omg_0, omg_1);
    }

    {
        let omg_2: R = omg[2];
        let omg_3: R = omg[3];
        cplx_twiddle(re_0, im_0, re_4, im_4, omg_2, omg_3);
        cplx_twiddle(re_1, im_1, re_5, im_5, omg_2, omg_3);
        cplx_twiddle(re_2, im_2, re_6, im_6, omg_2, omg_3);
        cplx_twiddle(re_3, im_3, re_7, im_7, omg_2, omg_3);

        cplx_i_twiddle(re_8, im_8, re_12, im_12, omg_2, omg_3);
        cplx_i_twiddle(re_9, im_9, re_13, im_13, omg_2, omg_3);
        cplx_i_twiddle(re_10, im_10, re_14, im_14, omg_2, omg_3);
        cplx_i_twiddle(re_11, im_11, re_15, im_15, omg_2, omg_3);
    }

    {
        let omg_0: R = omg[4];
        let omg_1: R = omg[5];
        let omg_2: R = omg[6];
        let omg_3: R = omg[7];
        cplx_twiddle(re_0, im_0, re_2, im_2, omg_0, omg_1);
        cplx_twiddle(re_1, im_1, re_3, im_3, omg_0, omg_1);
        cplx_twiddle(re_8, im_8, re_10, im_10, omg_2, omg_3);
        cplx_twiddle(re_9, im_9, re_11, im_11, omg_2, omg_3);

        cplx_i_twiddle(re_4, im_4, re_6, im_6, omg_0, omg_1);
        cplx_i_twiddle(re_5, im_5, re_7, im_7, omg_0, omg_1);
        cplx_i_twiddle(re_12, im_12, re_14, im_14, omg_2, omg_3);
        cplx_i_twiddle(re_13, im_13, re_15, im_15, omg_2, omg_3);
    }

    {
        let omg_0: R = omg[8];
        let omg_1: R = omg[9];
        let omg_2: R = omg[10];
        let omg_3: R = omg[11];
        let omg_4: R = omg[12];
        let omg_5: R = omg[13];
        let omg_6: R = omg[14];
        let omg_7: R = omg[15];
        cplx_twiddle(re_0, im_0, re_1, im_1, omg_0, omg_4);
        cplx_twiddle(re_4, im_4, re_5, im_5, omg_1, omg_5);
        cplx_twiddle(re_8, im_8, re_9, im_9, omg_2, omg_6);
        cplx_twiddle(re_12, im_12, re_13, im_13, omg_3, omg_7);

        cplx_i_twiddle(re_2, im_2, re_3, im_3, omg_0, omg_4);
        cplx_i_twiddle(re_6, im_6, re_7, im_7, omg_1, omg_5);
        cplx_i_twiddle(re_10, im_10, re_11, im_11, omg_2, omg_6);
        cplx_i_twiddle(re_14, im_14, re_15, im_15, omg_3, omg_7);
    }
}

#[inline(always)]
fn fft_bfs_16_ref<R: Float + FloatConst + Debug>(m: usize, re: &mut [R], im: &mut [R], omg: &[R], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm: usize = m;

    if !log_m.is_multiple_of(2) {
        let h: usize = mm >> 1;
        twiddle_fft_ref(h, re, im, as_arr::<2, R>(&omg[pos..]));
        pos += 2;
        mm = h
    }

    while mm > 16 {
        let h: usize = mm >> 2;
        for off in (0..m).step_by(mm) {
            bitwiddle_fft_ref(h, &mut re[off..], &mut im[off..], as_arr::<4, R>(&omg[pos..]));
            pos += 4;
        }
        mm = h
    }

    for off in (0..m).step_by(16) {
        fft16_ref(
            as_arr_mut::<16, R>(&mut re[off..]),
            as_arr_mut::<16, R>(&mut im[off..]),
            *as_arr::<16, R>(&omg[pos..]),
        );
        pos += 16;
    }

    pos
}

#[inline(always)]
fn twiddle_fft_ref<R: Float + FloatConst>(h: usize, re: &mut [R], im: &mut [R], omg: &[R; 2]) {
    let romg = omg[0];
    let iomg = omg[1];

    let (re_lhs, re_rhs) = re.split_at_mut(h);
    let (im_lhs, im_rhs) = im.split_at_mut(h);

    for i in 0..h {
        cplx_twiddle(&mut re_lhs[i], &mut im_lhs[i], &mut re_rhs[i], &mut im_rhs[i], romg, iomg);
    }
}

#[inline(always)]
fn bitwiddle_fft_ref<R: Float + FloatConst>(h: usize, re: &mut [R], im: &mut [R], omg: &[R; 4]) {
    let (r0, r2) = re.split_at_mut(2 * h);
    let (r0, r1) = r0.split_at_mut(h);
    let (r2, r3) = r2.split_at_mut(h);

    let (i0, i2) = im.split_at_mut(2 * h);
    let (i0, i1) = i0.split_at_mut(h);
    let (i2, i3) = i2.split_at_mut(h);

    let omg_0: R = omg[0];
    let omg_1: R = omg[1];
    let omg_2: R = omg[2];
    let omg_3: R = omg[3];

    for i in 0..h {
        cplx_twiddle(&mut r0[i], &mut i0[i], &mut r2[i], &mut i2[i], omg_0, omg_1);
        cplx_twiddle(&mut r1[i], &mut i1[i], &mut r3[i], &mut i3[i], omg_0, omg_1);
    }

    for i in 0..h {
        cplx_twiddle(&mut r0[i], &mut i0[i], &mut r1[i], &mut i1[i], omg_2, omg_3);
        cplx_i_twiddle(&mut r2[i], &mut i2[i], &mut r3[i], &mut i3[i], omg_2, omg_3);
    }
}
