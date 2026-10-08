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

use crate::kernels::fft64::reim::{as_arr, as_arr_mut};

/// Forward negacyclic FFT in reim layout, rounding every product.
#[inline(always)]
pub fn fft_portable<R: Float + FloatConst + Debug>(m: usize, omg: &[R], data: &mut [R]) {
    fft_portable_body::<R, false>(m, omg, data)
}

/// Forward negacyclic FFT in reim layout for CKKS encoding.
///
/// Every butterfly rounds one product and fuses the other into a
/// multiply-add, at the positions the AVX2, AVX-512 and NEON kernels use, so
/// all CPU backends encode to the same bits. On `x86_64` builds without the
/// `fma` target feature, the multiply-adds run on the FMA unit when the CPU
/// has one and in software otherwise, with the same results.
#[inline(always)]
pub fn fft_portable_fused<R: Float + FloatConst + Debug>(m: usize, omg: &[R], data: &mut [R]) {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports FMA.
        return unsafe { fft_portable_fma(m, omg, data) };
    }
    fft_portable_body::<R, true>(m, omg, data)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
#[target_feature(enable = "fma")]
fn fft_portable_fma<R: Float + FloatConst + Debug>(m: usize, omg: &[R], data: &mut [R]) {
    fft_portable_body::<R, true>(m, omg, data)
}

#[inline(always)]
fn fft_portable_body<R: Float + FloatConst + Debug, const FUSED: bool>(m: usize, omg: &[R], data: &mut [R]) {
    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m <= 16 {
        match m {
            1 => {}
            2 => fft2_portable::<_, FUSED>(as_arr_mut::<2, R>(re), as_arr_mut::<2, R>(im), *as_arr::<2, R>(omg)),
            4 => fft4_portable::<_, FUSED>(as_arr_mut::<4, R>(re), as_arr_mut::<4, R>(im), *as_arr::<4, R>(omg)),
            8 => fft8_portable::<_, FUSED>(as_arr_mut::<8, R>(re), as_arr_mut::<8, R>(im), *as_arr::<8, R>(omg)),
            16 => fft16_portable::<_, FUSED>(as_arr_mut::<16, R>(re), as_arr_mut::<16, R>(im), *as_arr::<16, R>(omg)),
            _ => {}
        }
    } else if m <= 2048 {
        fft_bfs_16_portable::<_, FUSED>(m, re, im, omg, 0);
    } else {
        fft_rec_16_portable::<_, FUSED>(m, re, im, omg, 0);
    }
}

/// Splits blocks larger than 2048 with one twiddle layer each, then runs the
/// breadth-first transform on every leaf, visiting the blocks in depth-first
/// preorder as the twiddle table lays them out. It is a loop rather than a
/// recursion so that it inlines into the FMA-enabled entry point.
#[inline(always)]
fn fft_rec_16_portable<R: Float + FloatConst + Debug, const FUSED: bool>(
    m: usize,
    re: &mut [R],
    im: &mut [R],
    omg: &[R],
    mut pos: usize,
) -> usize {
    // The stack holds at most one pending right sibling per level, plus the
    // current block.
    let mut blocks = [(0usize, 0usize); usize::BITS as usize + 1];
    blocks[0] = (0, m);
    let mut len = 1;
    while len > 0 {
        len -= 1;
        let (off, size) = blocks[len];
        let (re, im) = (&mut re[off..off + size], &mut im[off..off + size]);
        if size <= 2048 {
            pos = fft_bfs_16_portable::<_, FUSED>(size, re, im, omg, pos);
            continue;
        }
        let h = size >> 1;
        twiddle_fft_portable::<_, FUSED>(h, re, im, as_arr::<2, R>(&omg[pos..]));
        pos += 2;
        blocks[len] = (off + h, h);
        blocks[len + 1] = (off, h);
        len += 2;
    }
    pos
}

/// `(a, b) <- (a + b * w, a - b * w)`. `FUSED` rounds the products by `Im(w)`
/// and fuses the others.
#[inline(always)]
fn cplx_twiddle<R: Float + FloatConst, const FUSED: bool>(ra: &mut R, ia: &mut R, rb: &mut R, ib: &mut R, omg_re: R, omg_im: R) {
    let (dr, di): (R, R) = if FUSED {
        (rb.mul_add(omg_re, -(*ib * omg_im)), ib.mul_add(omg_re, *rb * omg_im))
    } else {
        (*rb * omg_re - *ib * omg_im, *rb * omg_im + *ib * omg_re)
    };
    *rb = *ra - dr;
    *ib = *ia - di;
    *ra = *ra + dr;
    *ia = *ia + di;
}

/// `(a, b) <- (a - b * i * w, a + b * i * w)`. `FUSED` rounds the products by
/// `Re(w)` and computes the imaginary part negated, as the SIMD kernels do,
/// which also fixes the sign of exact zeros.
#[inline(always)]
fn cplx_i_twiddle<R: Float + FloatConst, const FUSED: bool>(
    ra: &mut R,
    ia: &mut R,
    rb: &mut R,
    ib: &mut R,
    omg_re: R,
    omg_im: R,
) {
    let (dr, neg_di): (R, R) = if FUSED {
        (rb.mul_add(omg_im, *ib * omg_re), ib.mul_add(omg_im, -(*rb * omg_re)))
    } else {
        (*rb * omg_im + *ib * omg_re, -(*rb * omg_re - *ib * omg_im))
    };
    *rb = *ra + dr;
    *ib = *ia + neg_di;
    *ra = *ra - dr;
    *ia = *ia - neg_di;
}

#[inline(always)]
fn fft2_portable<R: Float + FloatConst, const FUSED: bool>(re: &mut [R; 2], im: &mut [R; 2], omg: [R; 2]) {
    let [ra, rb] = re;
    let [ia, ib] = im;
    let [romg, iomg] = omg;
    cplx_twiddle::<_, FUSED>(ra, ia, rb, ib, romg, iomg);
}

#[inline(always)]
fn fft4_portable<R: Float + FloatConst, const FUSED: bool>(re: &mut [R; 4], im: &mut [R; 4], omg: [R; 4]) {
    let [re_0, re_1, re_2, re_3] = re;
    let [im_0, im_1, im_2, im_3] = im;

    {
        let omg_0 = omg[0];
        let omg_1 = omg[1];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_2, im_2, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_3, im_3, omg_0, omg_1);
    }

    {
        let omg_0 = omg[2];
        let omg_1 = omg[3];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_1, im_1, omg_0, omg_1);
        cplx_i_twiddle::<_, FUSED>(re_2, im_2, re_3, im_3, omg_0, omg_1);
    }
}

#[inline(always)]
fn fft8_portable<R: Float + FloatConst, const FUSED: bool>(re: &mut [R; 8], im: &mut [R; 8], omg: [R; 8]) {
    let [re_0, re_1, re_2, re_3, re_4, re_5, re_6, re_7] = re;
    let [im_0, im_1, im_2, im_3, im_4, im_5, im_6, im_7] = im;

    {
        let omg_0 = omg[0];
        let omg_1 = omg[1];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_4, im_4, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_5, im_5, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_2, im_2, re_6, im_6, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_3, im_3, re_7, im_7, omg_0, omg_1);
    }

    {
        let omg_2 = omg[2];
        let omg_3 = omg[3];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_2, im_2, omg_2, omg_3);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_3, im_3, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_4, im_4, re_6, im_6, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_5, im_5, re_7, im_7, omg_2, omg_3);
    }

    {
        let omg_4 = omg[4];
        let omg_5 = omg[5];
        let omg_6 = omg[6];
        let omg_7 = omg[7];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_1, im_1, omg_4, omg_6);
        cplx_i_twiddle::<_, FUSED>(re_2, im_2, re_3, im_3, omg_4, omg_6);
        cplx_twiddle::<_, FUSED>(re_4, im_4, re_5, im_5, omg_5, omg_7);
        cplx_i_twiddle::<_, FUSED>(re_6, im_6, re_7, im_7, omg_5, omg_7);
    }
}

#[inline(always)]
fn fft16_portable<R: Float + FloatConst + Debug, const FUSED: bool>(re: &mut [R; 16], im: &mut [R; 16], omg: [R; 16]) {
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
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_8, im_8, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_9, im_9, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_2, im_2, re_10, im_10, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_3, im_3, re_11, im_11, omg_0, omg_1);

        cplx_twiddle::<_, FUSED>(re_4, im_4, re_12, im_12, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_5, im_5, re_13, im_13, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_6, im_6, re_14, im_14, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_7, im_7, re_15, im_15, omg_0, omg_1);
    }

    {
        let omg_2: R = omg[2];
        let omg_3: R = omg[3];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_4, im_4, omg_2, omg_3);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_5, im_5, omg_2, omg_3);
        cplx_twiddle::<_, FUSED>(re_2, im_2, re_6, im_6, omg_2, omg_3);
        cplx_twiddle::<_, FUSED>(re_3, im_3, re_7, im_7, omg_2, omg_3);

        cplx_i_twiddle::<_, FUSED>(re_8, im_8, re_12, im_12, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_9, im_9, re_13, im_13, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_10, im_10, re_14, im_14, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_11, im_11, re_15, im_15, omg_2, omg_3);
    }

    {
        let omg_0: R = omg[4];
        let omg_1: R = omg[5];
        let omg_2: R = omg[6];
        let omg_3: R = omg[7];
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_2, im_2, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_1, im_1, re_3, im_3, omg_0, omg_1);
        cplx_twiddle::<_, FUSED>(re_8, im_8, re_10, im_10, omg_2, omg_3);
        cplx_twiddle::<_, FUSED>(re_9, im_9, re_11, im_11, omg_2, omg_3);

        cplx_i_twiddle::<_, FUSED>(re_4, im_4, re_6, im_6, omg_0, omg_1);
        cplx_i_twiddle::<_, FUSED>(re_5, im_5, re_7, im_7, omg_0, omg_1);
        cplx_i_twiddle::<_, FUSED>(re_12, im_12, re_14, im_14, omg_2, omg_3);
        cplx_i_twiddle::<_, FUSED>(re_13, im_13, re_15, im_15, omg_2, omg_3);
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
        cplx_twiddle::<_, FUSED>(re_0, im_0, re_1, im_1, omg_0, omg_4);
        cplx_twiddle::<_, FUSED>(re_4, im_4, re_5, im_5, omg_1, omg_5);
        cplx_twiddle::<_, FUSED>(re_8, im_8, re_9, im_9, omg_2, omg_6);
        cplx_twiddle::<_, FUSED>(re_12, im_12, re_13, im_13, omg_3, omg_7);

        cplx_i_twiddle::<_, FUSED>(re_2, im_2, re_3, im_3, omg_0, omg_4);
        cplx_i_twiddle::<_, FUSED>(re_6, im_6, re_7, im_7, omg_1, omg_5);
        cplx_i_twiddle::<_, FUSED>(re_10, im_10, re_11, im_11, omg_2, omg_6);
        cplx_i_twiddle::<_, FUSED>(re_14, im_14, re_15, im_15, omg_3, omg_7);
    }
}

#[inline(always)]
fn fft_bfs_16_portable<R: Float + FloatConst + Debug, const FUSED: bool>(
    m: usize,
    re: &mut [R],
    im: &mut [R],
    omg: &[R],
    mut pos: usize,
) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm: usize = m;

    if !log_m.is_multiple_of(2) {
        let h: usize = mm >> 1;
        twiddle_fft_portable::<_, FUSED>(h, re, im, as_arr::<2, R>(&omg[pos..]));
        pos += 2;
        mm = h
    }

    while mm > 16 {
        let h: usize = mm >> 2;
        for off in (0..m).step_by(mm) {
            bitwiddle_fft_portable::<_, FUSED>(h, &mut re[off..], &mut im[off..], as_arr::<4, R>(&omg[pos..]));
            pos += 4;
        }
        mm = h
    }

    for off in (0..m).step_by(16) {
        fft16_portable::<_, FUSED>(
            as_arr_mut::<16, R>(&mut re[off..]),
            as_arr_mut::<16, R>(&mut im[off..]),
            *as_arr::<16, R>(&omg[pos..]),
        );
        pos += 16;
    }

    pos
}

/// Elements a pass loads, combines and stores as a unit.
///
/// Working on local copies is what lets a compiler use vector instructions without proving that the
/// quarters of a block do not overlap. Every element goes through the same operations in the same order
/// as in an element by element pass, so the results are bit-identical.
const LANES: usize = 4;

#[inline(always)]
fn load<R: Float>(x: &[R]) -> [R; LANES] {
    [x[0], x[1], x[2], x[3]]
}

#[inline(always)]
fn twiddle_fft_portable<R: Float + FloatConst, const FUSED: bool>(h: usize, re: &mut [R], im: &mut [R], omg: &[R; 2]) {
    let romg = omg[0];
    let iomg = omg[1];

    let (re_lhs, re_rhs) = re.split_at_mut(h);
    let (im_lhs, im_rhs) = im.split_at_mut(h);

    if h < LANES {
        for i in 0..h {
            cplx_twiddle::<_, FUSED>(&mut re_lhs[i], &mut im_lhs[i], &mut re_rhs[i], &mut im_rhs[i], romg, iomg);
        }
        return;
    }
    for (((ra, ia), rb), ib) in re_lhs[..h]
        .chunks_exact_mut(LANES)
        .zip(im_lhs[..h].chunks_exact_mut(LANES))
        .zip(re_rhs[..h].chunks_exact_mut(LANES))
        .zip(im_rhs[..h].chunks_exact_mut(LANES))
    {
        let (mut va, mut wa, mut vb, mut wb) = (load(ra), load(ia), load(rb), load(ib));
        for l in 0..LANES {
            cplx_twiddle::<_, FUSED>(&mut va[l], &mut wa[l], &mut vb[l], &mut wb[l], romg, iomg);
        }
        ra.copy_from_slice(&va);
        ia.copy_from_slice(&wa);
        rb.copy_from_slice(&vb);
        ib.copy_from_slice(&wb);
    }
}

/// Two merged layers over a block of `4 * h` elements, `h` a multiple of [`LANES`].
#[inline(always)]
fn bitwiddle_fft_portable<R: Float + FloatConst, const FUSED: bool>(h: usize, re: &mut [R], im: &mut [R], omg: &[R; 4]) {
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

    let re = r0
        .chunks_exact_mut(LANES)
        .zip(r1.chunks_exact_mut(LANES))
        .zip(r2[..h].chunks_exact_mut(LANES))
        .zip(r3[..h].chunks_exact_mut(LANES));
    let im = i0
        .chunks_exact_mut(LANES)
        .zip(i1.chunks_exact_mut(LANES))
        .zip(i2[..h].chunks_exact_mut(LANES))
        .zip(i3[..h].chunks_exact_mut(LANES));
    for ((((r0, r1), r2), r3), (((i0, i1), i2), i3)) in re.zip(im) {
        let (mut a0, mut a1, mut a2, mut a3) = (load(r0), load(r1), load(r2), load(r3));
        let (mut b0, mut b1, mut b2, mut b3) = (load(i0), load(i1), load(i2), load(i3));
        for l in 0..LANES {
            cplx_twiddle::<_, FUSED>(&mut a0[l], &mut b0[l], &mut a2[l], &mut b2[l], omg_0, omg_1);
            cplx_twiddle::<_, FUSED>(&mut a1[l], &mut b1[l], &mut a3[l], &mut b3[l], omg_0, omg_1);
        }
        for l in 0..LANES {
            cplx_twiddle::<_, FUSED>(&mut a0[l], &mut b0[l], &mut a1[l], &mut b1[l], omg_2, omg_3);
            cplx_i_twiddle::<_, FUSED>(&mut a2[l], &mut b2[l], &mut a3[l], &mut b3[l], omg_2, omg_3);
        }
        r0.copy_from_slice(&a0);
        r1.copy_from_slice(&a1);
        r2.copy_from_slice(&a2);
        r3.copy_from_slice(&a3);
        i0.copy_from_slice(&b0);
        i1.copy_from_slice(&b1);
        i2.copy_from_slice(&b2);
        i3.copy_from_slice(&b3);
    }
}
