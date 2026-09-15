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

use crate::reference::fft64::reim::{ReimFFTExecute, frac_rev_bits, ifft_ref::ifft_ref};
use poulpy_hal::alloc_aligned;

pub struct ReimIFFTRef;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTRef {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_ref(table.m, &table.omg, data);
    }
}

pub struct ReimIFFTTable<R: Float + FloatConst + Debug> {
    m: usize,
    omg: Vec<R>,
}

impl<R: Float + FloatConst + Debug> ReimIFFTTable<R> {
    pub fn new(m: usize) -> Self {
        assert!(m & (m - 1) == 0, "m must be a power of two but is {m}");
        let mut omg: Vec<R> = alloc_aligned::<R>(2 * m);

        let quarter: R = R::exp2(R::from(-2).unwrap());

        if m <= 16 {
            match m {
                1 => {}
                2 => {
                    fill_ifft2_omegas::<R>(quarter, &mut omg, 0);
                }
                4 => {
                    fill_ifft4_omegas(quarter, &mut omg, 0);
                }
                8 => {
                    fill_ifft8_omegas(quarter, &mut omg, 0);
                }
                16 => {
                    fill_ifft16_omegas(quarter, &mut omg, 0);
                }
                _ => {}
            }
        } else if m <= 2048 {
            fill_ifft_bfs_16_omegas(m, quarter, &mut omg, 0);
        } else {
            fill_ifft_rec_16_omegas(m, quarter, &mut omg, 0);
        }

        Self { m, omg }
    }

    pub fn execute(&self, data: &mut [R]) {
        ifft_ref(self.m, &self.omg, data);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn omg(&self) -> &[R] {
        &self.omg
    }
}

#[inline(always)]
fn fill_ifft2_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 2);
    let angle: R = j / R::exp2(R::from(2).unwrap());
    let two_pi: R = R::exp2(R::from(2).unwrap()) * R::PI();
    omg_pos[0] = R::cos(two_pi * angle);
    omg_pos[1] = -R::sin(two_pi * angle);
    pos + 2
}

#[inline(always)]
fn fill_ifft4_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 4);
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let two_pi: R = R::from(2).unwrap() * R::PI();
    omg_pos[0] = R::cos(two_pi * angle_2);
    omg_pos[1] = -R::sin(two_pi * angle_2);
    omg_pos[2] = R::cos(two_pi * angle_1);
    omg_pos[3] = -R::sin(two_pi * angle_1);
    pos + 4
}

#[inline(always)]
fn fill_ifft8_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 8);
    let _8th: R = R::from(1. / 8.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    let two_pi: R = R::from(2).unwrap() * R::PI();
    omg_pos[0] = R::cos(two_pi * angle_4);
    omg_pos[1] = R::cos(two_pi * (angle_4 + _8th));
    omg_pos[2] = -R::sin(two_pi * angle_4);
    omg_pos[3] = -R::sin(two_pi * (angle_4 + _8th));
    omg_pos[4] = R::cos(two_pi * angle_2);
    omg_pos[5] = -R::sin(two_pi * angle_2);
    omg_pos[6] = R::cos(two_pi * angle_1);
    omg_pos[7] = -R::sin(two_pi * angle_1);
    pos + 8
}

#[inline(always)]
fn fill_ifft16_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 16);
    let _8th: R = R::from(1. / 8.).unwrap();
    let _16th: R = R::from(1. / 16.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    let angle_8: R = j / R::from(16).unwrap();
    let two_pi: R = R::from(2).unwrap() * R::PI();
    omg_pos[0] = R::cos(two_pi * angle_8);
    omg_pos[1] = R::cos(two_pi * (angle_8 + _8th));
    omg_pos[2] = R::cos(two_pi * (angle_8 + _16th));
    omg_pos[3] = R::cos(two_pi * (angle_8 + _8th + _16th));
    omg_pos[4] = -R::sin(two_pi * angle_8);
    omg_pos[5] = -R::sin(two_pi * (angle_8 + _8th));
    omg_pos[6] = -R::sin(two_pi * (angle_8 + _16th));
    omg_pos[7] = -R::sin(two_pi * (angle_8 + _8th + _16th));
    omg_pos[8] = R::cos(two_pi * angle_4);
    omg_pos[9] = -R::sin(two_pi * angle_4);
    omg_pos[10] = R::cos(two_pi * (angle_4 + _8th));
    omg_pos[11] = -R::sin(two_pi * (angle_4 + _8th));
    omg_pos[12] = R::cos(two_pi * angle_2);
    omg_pos[13] = -R::sin(two_pi * angle_2);
    omg_pos[14] = R::cos(two_pi * angle_1);
    omg_pos[15] = -R::sin(two_pi * angle_1);
    pos + 16
}

#[inline(always)]
fn fill_ifft_bfs_16_omegas<R: Float + FloatConst + Debug>(m: usize, j: R, omg: &mut [R], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut jj: R = j * R::from(16).unwrap() / R::from(m).unwrap();

    for i in (0..m).step_by(16) {
        let j = jj + frac_rev_bits(i >> 4);
        fill_ifft16_omegas(j, omg, pos);
        pos += 16
    }

    let mut h: usize = 16;
    let m_half: usize = m >> 1;

    let two_pi: R = R::from(2).unwrap() * R::PI();

    while h < m_half {
        let mm: usize = h << 2;
        for i in (0..m).step_by(mm) {
            let rs_0 = jj + frac_rev_bits::<R>(i / mm) / R::from(4).unwrap();
            let rs_1 = R::from(2).unwrap() * rs_0;
            omg[pos] = R::cos(two_pi * rs_0);
            omg[pos + 1] = -R::sin(two_pi * rs_0);
            omg[pos + 2] = R::cos(two_pi * rs_1);
            omg[pos + 3] = -R::sin(two_pi * rs_1);
            pos += 4;
        }
        h = mm;
        jj = jj * R::from(4).unwrap();
    }

    if !log_m.is_multiple_of(2) {
        omg[pos] = R::cos(two_pi * jj);
        omg[pos + 1] = -R::sin(two_pi * jj);
        pos += 2;
        jj = jj * R::from(2).unwrap();
    }

    assert_eq!(jj, j);

    pos
}

#[inline(always)]
fn fill_ifft_rec_16_omegas<R: Float + FloatConst + Debug>(m: usize, j: R, omg: &mut [R], mut pos: usize) -> usize {
    if m <= 2048 {
        return fill_ifft_bfs_16_omegas(m, j, omg, pos);
    }
    let h: usize = m >> 1;
    let s: R = j / R::from(2).unwrap();
    pos = fill_ifft_rec_16_omegas(h, s, omg, pos);
    pos = fill_ifft_rec_16_omegas(h, s + R::from(0.5).unwrap(), omg, pos);
    let _2pi = R::from(2).unwrap() * R::PI();
    omg[pos] = R::cos(_2pi * s);
    omg[pos + 1] = -R::sin(_2pi * s);
    pos += 2;
    pos
}
