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

use crate::reference::fft64::reim::{ReimFFTExecute, fft_ref, frac_rev_bits, scalar_root};
use poulpy_hal::alloc_aligned;

pub struct ReimFFTRef;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTRef {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_ref(table.m, &table.omg, data);
    }
}

pub struct ReimFFTTable<R: Float + FloatConst + Debug> {
    m: usize,
    omg: Vec<R>,
}

impl<R: Float + FloatConst + Debug> ReimFFTTable<R> {
    pub fn new(m: usize) -> Self {
        Self::new_with_roots(m, scalar_root)
    }

    /// Builds the same layout from `root(t) = (cos 2πt, sin 2πt)`, where each
    /// turn `t` is an exact multiple of `1 / (4m)`.
    pub fn new_with_roots(m: usize, root: impl Fn(R) -> (R, R) + Copy) -> Self {
        assert!(m & (m - 1) == 0, "m must be a power of two but is {m}");
        let mut omg: Vec<R> = alloc_aligned::<R>(2 * m);

        let quarter: R = R::from(1. / 4.).unwrap();

        if m <= 16 {
            match m {
                1 => {}
                2 => {
                    fill_fft2_omegas(quarter, &mut omg, 0, root);
                }
                4 => {
                    fill_fft4_omegas(quarter, &mut omg, 0, root);
                }
                8 => {
                    fill_fft8_omegas(quarter, &mut omg, 0, root);
                }
                16 => {
                    fill_fft16_omegas(quarter, &mut omg, 0, root);
                }
                _ => {}
            }
        } else if m <= 2048 {
            fill_fft_bfs_16_omegas(m, quarter, &mut omg, 0, root);
        } else {
            fill_fft_rec_16_omegas(m, quarter, &mut omg, 0, root);
        }

        Self { m, omg }
    }

    pub fn execute(&self, data: &mut [R]) {
        fft_ref(self.m, &self.omg, data);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn omg(&self) -> &[R] {
        &self.omg
    }
}

#[inline(always)]
fn fill_fft2_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 2);
    let angle: R = j / R::from(2).unwrap();
    (omg_pos[0], omg_pos[1]) = root(angle);
    pos + 2
}

#[inline(always)]
fn fill_fft4_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 4);
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    (omg_pos[0], omg_pos[1]) = root(angle_1);
    (omg_pos[2], omg_pos[3]) = root(angle_2);
    pos + 4
}

#[inline(always)]
fn fill_fft8_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 8);
    let _8th: R = R::from(1. / 8.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    (omg_pos[0], omg_pos[1]) = root(angle_1);
    (omg_pos[2], omg_pos[3]) = root(angle_2);
    (omg_pos[4], omg_pos[6]) = root(angle_4);
    (omg_pos[5], omg_pos[7]) = root(angle_4 + _8th);
    pos + 8
}

#[inline(always)]
fn fill_fft16_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 16);
    let _8th: R = R::from(1. / 8.).unwrap();
    let _16th: R = R::from(1. / 16.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    let angle_8: R = j / R::from(16).unwrap();
    (omg_pos[0], omg_pos[1]) = root(angle_1);
    (omg_pos[2], omg_pos[3]) = root(angle_2);
    (omg_pos[4], omg_pos[5]) = root(angle_4);
    (omg_pos[6], omg_pos[7]) = root(angle_4 + _8th);
    (omg_pos[8], omg_pos[12]) = root(angle_8);
    (omg_pos[9], omg_pos[13]) = root(angle_8 + _8th);
    (omg_pos[10], omg_pos[14]) = root(angle_8 + _16th);
    (omg_pos[11], omg_pos[15]) = root(angle_8 + _8th + _16th);
    pos + 16
}

#[inline(always)]
fn fill_fft_bfs_16_omegas<R: Float + FloatConst>(
    m: usize,
    j: R,
    omg: &mut [R],
    mut pos: usize,
    root: impl Fn(R) -> (R, R) + Copy,
) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm: usize = m;
    let mut jj: R = j;

    if !log_m.is_multiple_of(2) {
        let h = mm >> 1;
        let j: R = jj * R::from(0.5).unwrap();
        (omg[pos], omg[pos + 1]) = root(j);
        pos += 2;
        mm = h;
        jj = j
    }

    while mm > 16 {
        let h: usize = mm >> 2;
        let j: R = jj * R::from(1. / 4.).unwrap();
        for i in (0..m).step_by(mm) {
            let rs_0 = j + frac_rev_bits::<R>(i / mm) * R::from(1. / 4.).unwrap();
            let rs_1 = R::from(2).unwrap() * rs_0;
            (omg[pos], omg[pos + 1]) = root(rs_1);
            (omg[pos + 2], omg[pos + 3]) = root(rs_0);
            pos += 4;
        }
        mm = h;
        jj = j;
    }

    for i in (0..m).step_by(16) {
        let j = jj + frac_rev_bits(i >> 4);
        fill_fft16_omegas(j, omg, pos, root);
        pos += 16
    }

    pos
}

#[inline(always)]
fn fill_fft_rec_16_omegas<R: Float + FloatConst>(
    m: usize,
    j: R,
    omg: &mut [R],
    mut pos: usize,
    root: impl Fn(R) -> (R, R) + Copy,
) -> usize {
    if m <= 2048 {
        return fill_fft_bfs_16_omegas(m, j, omg, pos, root);
    }
    let h: usize = m >> 1;
    let s: R = j * R::from(0.5).unwrap();
    (omg[pos], omg[pos + 1]) = root(s);
    pos += 2;
    pos = fill_fft_rec_16_omegas(h, s, omg, pos, root);
    pos = fill_fft_rec_16_omegas(h, s + R::from(0.5).unwrap(), omg, pos, root);
    pos
}

#[inline(always)]
#[allow(dead_code)]
fn ctwiddle_ref(ra: &mut f64, ia: &mut f64, rb: &mut f64, ib: &mut f64, omg_re: f64, omg_im: f64) {
    let dr: f64 = *rb * omg_re - *ib * omg_im;
    let di: f64 = *rb * omg_im + *ib * omg_re;
    *rb = *ra - dr;
    *ib = *ia - di;
    *ra += dr;
    *ia += di;
}
